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
// destructor-based save() in ~Cache() is unreliable. Each saveLocked()
// rewrites the whole file, so the threshold balances I/O cost against the
// risk of losing entries that were added but not yet persisted.
static constexpr size_t SAVE_THRESHOLD = 8;

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
    // does not invoke static destructors), so the periodic saves performed
    // by putByHash() during runtime are the primary persistence mechanism.
    if (dirty) save();
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
    ++pendingWrites;

    // Periodically persist the cache to disk so compiled shaders survive
    // process termination (e.g. SIGKILL on Android, where ~Cache() never
    // runs). Writing on every put would be too expensive — the entire
    // cache is rewritten each time — so writes are batched. saveLocked()
    // reuses the lock already held here (save() would deadlock).
    if (pendingWrites >= SAVE_THRESHOLD) {
        saveLocked();
    }
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

void Cache::saveLocked() {
    // Assumes cacheMutex is held by the caller.
    if (global_settings.max_glsl_cache_size <= 0) return;

    // Write to a temporary file first, then atomically rename it over the
    // real cache file. This guarantees glsl_cache.tmp is never left in a
    // partially-written (corrupt) state if the process is killed mid-write
    // (e.g. SIGKILL on Android). A corrupt file would cause load() to
    // discard the entire cache on the next launch, defeating the purpose.
    string tmpPath = string(glsl_cache_file_path) + ".write_tmp";
    ofstream file(tmpPath, ios::binary | ios::trunc);
    if (!file) return;

    size_t count = cacheList.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (const auto& entry : cacheList) {
        file.write(reinterpret_cast<const char*>(entry.sha256.data()), (long)entry.sha256.size());
        file.write(reinterpret_cast<const char*>(&entry.return_code), sizeof(entry.return_code));
        size_t esslSize = entry.size;
        file.write(reinterpret_cast<const char*>(&esslSize), sizeof(esslSize));
        file.write(entry.essl.data(), (long)esslSize);
    }

    file.flush();
    file.close();

    // Only swap in the new file if the write completed successfully;
    // otherwise keep the existing cache file intact.
    if (!file) {
        LOG_E("Failed to write shader cache temp file: %s", tmpPath.c_str())
        return;
    }

    if (rename(tmpPath.c_str(), glsl_cache_file_path) != 0) {
        LOG_E("Failed to atomically rename shader cache file: %s -> %s",
              tmpPath.c_str(), glsl_cache_file_path)
        return;
    }

    dirty = false;
    pendingWrites = 0;
}

void Cache::save() {
    std::lock_guard<std::mutex> lock(cacheMutex);
    saveLocked();
}

Cache& Cache::get_instance() {
    static Cache s_cache;
    return s_cache;
}
