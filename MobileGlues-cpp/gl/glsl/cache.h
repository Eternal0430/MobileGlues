// MobileGlues - gl/glsl/cache.h
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

#ifndef MOBILEGLUES_PLUGIN_CACHE_H
#define MOBILEGLUES_PLUGIN_CACHE_H

#include "../mg.h"
#include "../../config/config.h"
#include "../../config/settings.h"

#include <list>
#include <array>
#include <string>
#include <cstdint>
#include <mutex>

class Cache {
public:
    Cache();
    ~Cache();
    const char* get(const char* glsl, int* return_code = nullptr);
    void put(const char* glsl, const char* essl, int return_code = 0);

    // Lookup/store by pre-computed hash (avoids key string construction on cache hit).
    const char* getByHash(const std::array<uint8_t, 32>& hash, int* return_code = nullptr);
    void putByHash(const std::array<uint8_t, 32>& hash, const char* essl, int return_code = 0);

    bool load();
    void save();

    static Cache& get_instance();

    // Compute FNV-1a hash incrementally over two concatenated null-terminated strings,
    // producing the same result as computeSHA256(a+b) without building the concatenated string.
    static std::array<uint8_t, 32> computeSHA256Parts(const char* a, const char* b);

private:
    struct CacheEntry {
        std::array<uint8_t, 32> sha256;
        std::string essl;
        size_t size;
        int return_code = 0;
    };

    struct SHA256Hash {
        size_t operator()(const std::array<uint8_t, 32>& key) const;
    };

    std::list<CacheEntry> cacheList;
    using ListIterator = std::list<CacheEntry>::iterator;
    UnorderedMap<std::array<uint8_t, 32>, ListIterator, SHA256Hash> cacheMap;
    size_t cacheSize = 0;
    bool dirty = false;
    size_t pendingWrites = 0;  // new entries added since the last saveLocked()

    // Guards cacheList, cacheMap, cacheSize, dirty, pendingWrites.
    // Taken in getByHash/putByHash (runtime hot paths) and in save().
    // load() runs inside the constructor (single-threaded, before the
    // singleton is published) and does not take the lock; its catch block
    // calls save() which acquires the lock — safe because load() does not
    // hold it. saveLocked() assumes the lock is already held and is used
    // by putByHash() to persist periodically without re-locking.
    std::mutex cacheMutex;

    static std::array<uint8_t, 32> computeSHA256(const char* data);
    void maintainCacheSize();
    void saveLocked();  // assumes cacheMutex is held
};

#endif
