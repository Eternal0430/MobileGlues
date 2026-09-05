// MobileGlues - gl/glsl/cache.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

#include "cache.h"
#include "../log.h"
#include <fstream>
#include <cstring>
#include <cstdio>
#include <vector>

using namespace std;

// Persist the cache to disk after this many new entries, so compiled
// shaders survive process termination. On Android the process is typically
// killed with SIGKILL, which never runs static destructors — so the
// destructor-based save() in ~Cache() is unreliable. A background thread
// rewrites the whole file, so the threshold balances I/O cost against the
// risk of losing entries that were added but not yet persisted.

// Fast non-cryptographic hash (FNV-1a 64-bit) for cache keys.
// Stored in the first 8 bytes of the 32-byte array; remaining bytes are zero.
// ~10x faster than the previous SHA-256 implementation while providing
// sufficient collision resistance for a shader cache.
array<uint8_t, 32> Cache::computeSHA256(const char* data) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
    for (; *p; ++p) {
        hash ^= *p;
        hash *= 0x100000001b3ULL;
    }
    array<uint8_t, 32> result{};
    for (int i = 0; i < 8; ++i) {
        result[i] = static_cast<uint8_t>(hash >> (i * 8));
    }
    return result;
}

size_t Cache::SHA256Hash::operator()(const array<uint8_t, 32>& key) const {
    // Use the first 8 bytes (FNV-1a result) as a uint64_t.
    // On 64-bit platforms this is a perfect hash for the bucket index.
    uint64_t h = 0;
    for (int i = 0; i < 8; ++i) {
        h |= static_cast<uint64_t>(key[i]) << (i * 8);
    }
    return static_cast<size_t>(h);
}

// Incremental FNV-1a over two concatenated null-terminated strings.
// Produces the same hash as computeSHA256 of the concatenated string,
// but avoids building a temporary std::string (saves a full copy of `a`).
array<uint8_t, 32> Cache::computeSHA256Parts(const char* a, const char* b) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(a);
    for (; *p; ++p) {
        hash ^= *p;
        hash *= 0x100000001b3ULL;
    }
    p = reinterpret_cast<const uint8_t*>(b);
    for (; *p; ++p) {
        hash ^= *p;
        hash *= 0x100000001b3ULL;
    }
    array<uint8_t, 32> result{};
    for (int i = 0; i < 8; ++i) {
        result[i] = static_cast<uint8_t>(hash >> (i * 8));
    }
    return result;
}

Cache::Cache() {
    load();
}

Cache::~Cache() {
    // Persist any remaining dirty entries at graceful process exit. On
    // Android this rarely runs (processes are killed with SIGKILL, which
    // does not invoke static destructors), so the background thread started
    // by putByHash() is the primary persistence mechanism — it saves once
    // compilation goes quiet, rather than on a fixed entry count.
    stop_save_thread.store(true, std::memory_order_release);
    save();
}

const char* Cache::get(const char* glsl, int* return_code) {
    if (global_settings.max_glsl_cache_size <= 0) return nullptr;
    auto hash = computeSHA256(glsl);
    return getByHash(hash, return_code);
}

const char* Cache::getByHash(const std::array<uint8_t, 32>& hash, int* return_code) {
    if (global_settings.max_glsl_cache_size <= 0) return nullptr;
    std::lock_guard<std::mutex> lock(cacheMutex);
    auto it = cacheMap.find(hash);
    if (it == cacheMap.end()) return nullptr;

    cacheList.splice(cacheList.end(), cacheList, it->second);
    if (return_code) *return_code = it->second->return_code;
    return it->second->essl.c_str();
}

void Cache::put(const char* glsl, const char* essl, int return_code) {
    if (global_settings.max_glsl_cache_size <= 0) return;
    auto hash = computeSHA256(glsl);
    putByHash(hash, essl, return_code);
}

void Cache::putByHash(const std::array<uint8_t, 32>& hash, const char* essl, int return_code) {
    if (global_settings.max_glsl_cache_size <= 0) return;
    size_t esslStrSize = strlen(essl) + 1;

    size_t entryMemory = sizeof(CacheEntry::sha256) + sizeof(size_t) + sizeof(int) + esslStrSize;

    std::lock_guard<std::mutex> lock(cacheMutex);

    if (auto it = cacheMap.find(hash); it != cacheMap.end()) {
        cacheSize -= (sizeof(CacheEntry::sha256) + sizeof(size_t) + sizeof(int) + it->second->size);
        cacheList.erase(it->second);
        cacheMap.erase(it);
    }

    cacheList.emplace_back(CacheEntry{hash, essl, esslStrSize, return_code});
    cacheMap[hash] = prev(cacheList.end());
    cacheSize += entryMemory;

    maintainCacheSize();
    dirty = true;

    // Record when the last entry arrived, then let the background thread
    // decide when to write.
    //
    // This used to call saveLocked() every SAVE_THRESHOLD (8) new entries.
    // That was quadratic: saveLocked() rewrites the ENTIRE cache file, so
    // saving after every 8 new entries during a run that compiles N shaders
    // writes roughly N²/16 entries' worth of data in total. With a cache
    // budget of 710 MB (the configured maxGlslCacheSize) and a cold start
    // compiling a few thousand shaders, that is multiple gigabytes written
    // to flash — synchronously, while HOLDING cacheMutex, so every other
    // compiling thread blocks behind it.
    //
    // The measured symptom matched exactly: hundreds of thousands of GL
    // calls at a few thousand per second for many minutes, with
    // eglSwapBuffers never once called, because the render thread sits in
    // join() waiting for the compile executor to finish.
    //
    // Deferring to a quiet moment means a cold start writes the cache once,
    // after compilation finishes, instead of hundreds of times during it.
    last_put_time.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                        std::memory_order_release);
    MaybeStartSaveThread();
}

void Cache::MaybeStartSaveThread() {
    if (stop_save_thread.load(std::memory_order_acquire)) return;
    if (save_thread_started.exchange(true, std::memory_order_acq_rel)) return;
    std::thread(&Cache::SaveThreadLoop, this).detach();
}

void Cache::SaveThreadLoop() {
    // Save once compilation has gone quiet for kIdleBeforeSave, rather than
    // while it is still streaming. A cold start compiles thousands of
    // shaders back to back; writing the file after each batch would be
    // wasted work, since the next batch would immediately invalidate it.
    constexpr auto kIdleBeforeSave = std::chrono::seconds(3);
    constexpr auto kPollInterval = std::chrono::seconds(2);
    // Upper bound on how long new entries may sit unsaved. A compile storm
    // that never goes quiet (a very long first run) would otherwise lose
    // everything if the process is killed, since nothing would ever be
    // written until it ended.
    constexpr auto kMaxSaveInterval = std::chrono::seconds(60);

    auto last_save = std::chrono::steady_clock::now();

    while (!stop_save_thread.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(kPollInterval);

        const auto now = std::chrono::steady_clock::now();
        const auto last = std::chrono::steady_clock::time_point(
            std::chrono::steady_clock::duration(last_put_time.load(std::memory_order_acquire)));
        if (last.time_since_epoch().count() == 0) continue;

        const bool quiet = (now - last) >= kIdleBeforeSave;
        const bool overdue = (now - last_save) >= kMaxSaveInterval;
        if (!quiet && !overdue) continue;

        bool was_dirty = false;
        {
            std::lock_guard<std::mutex> lock(cacheMutex);
            was_dirty = dirty;
        }
        if (was_dirty) FlushToDisk();
        last_save = std::chrono::steady_clock::now();
    }

    // Final flush on the way out.
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        if (dirty) FlushToDisk();
    }
}

void Cache::FlushToDisk() {
    // Copy under the lock, write after releasing it.
    std::vector<CacheEntry> snapshot;
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        snapshot = SnapshotLocked();
    }
    if (snapshot.empty()) return;

    WriteSnapshot(snapshot);

    std::lock_guard<std::mutex> lock(cacheMutex);
    dirty = false;
}

void Cache::WriteSnapshot(const std::vector<CacheEntry>& snapshot) {
    if (global_settings.max_glsl_cache_size <= 0) return;

    // Write to a temporary file first, then atomically rename it over the
    // real cache file, so a process killed mid-write cannot leave a
    // truncated cache behind.
    const std::string tmpPath = std::string(glsl_cache_file_path) + ".write_tmp";
    ofstream file(tmpPath, ios::binary | ios::trunc);
    if (!file) return;

    const size_t count = snapshot.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (const auto& entry : snapshot) {
        file.write(reinterpret_cast<const char*>(entry.sha256.data()), (long)entry.sha256.size());
        file.write(reinterpret_cast<const char*>(&entry.return_code), sizeof(entry.return_code));
        const size_t esslSize = entry.size;
        file.write(reinterpret_cast<const char*>(&esslSize), sizeof(esslSize));
        file.write(entry.essl.data(), (long)esslSize);
    }

    file.flush();
    file.close();

    if (!file) {
        LOG_E("Failed to write shader cache temp file: %s", tmpPath.c_str())
        return;
    }

    if (rename(tmpPath.c_str(), glsl_cache_file_path) != 0) {
        LOG_E("Failed to atomically rename shader cache file: %s -> %s",
              tmpPath.c_str(), glsl_cache_file_path)
        return;
    }

    LOG_V("Wrote shader cache: %zu entries", snapshot.size());
}

std::vector<Cache::CacheEntry> Cache::SnapshotLocked() const {
    std::vector<CacheEntry> snapshot;
    snapshot.reserve(cacheList.size());
    for (const auto& entry : cacheList) snapshot.push_back(entry);
    return snapshot;
}

void Cache::maintainCacheSize() {
    if (global_settings.max_glsl_cache_size <= 0) return;
    while (cacheSize > global_settings.max_glsl_cache_size && !cacheList.empty()) {
        const auto& oldEntry = cacheList.front();
        size_t removedMemory = sizeof(CacheEntry::sha256) + sizeof(size_t) + sizeof(int) + oldEntry.size;
        cacheSize -= removedMemory;
        cacheMap.erase(oldEntry.sha256);
        cacheList.pop_front();
    }
}

bool Cache::load() {
    try {
        ifstream file(glsl_cache_file_path, ios::binary);
        if (!file) return false;

        size_t count;
        file.read(reinterpret_cast<char*>(&count), sizeof(count));

        while (count--) {
            array<uint8_t, 32> hash{};
            int return_code = 0;
            size_t esslSize;

            file.read(reinterpret_cast<char*>(hash.data()), hash.size());
            file.read(reinterpret_cast<char*>(&return_code), sizeof(return_code));
            file.read(reinterpret_cast<char*>(&esslSize), sizeof(esslSize));

            string essl(esslSize, '\0');
            file.read(essl.data(), (long)esslSize);

            if (cacheMap.count(hash)) continue;

            size_t entryMemory = sizeof(CacheEntry::sha256) + sizeof(size_t) + sizeof(int) + esslSize;
            cacheSize += entryMemory;

            cacheList.emplace_back(CacheEntry{hash, move(essl), esslSize, return_code});
            cacheMap[hash] = prev(cacheList.end());
        }

        maintainCacheSize();
        return true;
    }
    catch (...) {
        LOG_W_FORCE("Error while loading glsl cache file. Clearing it...")
        cacheMap.clear();
        cacheSize = 0;
        cacheList.clear();
        save();
        return false;
    }
}


void Cache::save() {
    FlushToDisk();
}

Cache& Cache::get_instance() {
    static Cache s_cache;
    return s_cache;
}
