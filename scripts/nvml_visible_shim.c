#define _GNU_SOURCE

#include <dlfcn.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int nvmlReturn_t;
typedef struct nvmlDevice_st *nvmlDevice_t;

#define NVML_SUCCESS 0
#define NVML_ERROR_INVALID_ARGUMENT 2

typedef nvmlReturn_t (*nvml_count_fn)(unsigned int *);
typedef nvmlReturn_t (*nvml_handle_fn)(unsigned int, nvmlDevice_t *);

static int g_ids[64];
static size_t g_id_count = 0;
static int g_initialized = 0;
static void *g_nvml_handle = NULL;
static int g_use_physical_span = 0;

static void ensure_visible_ids(void) {
    if (g_initialized) {
        return;
    }

    g_initialized = 1;

    const char *raw = getenv("GPTQMODEL_VLLM_HEALTHY_PHYSICAL_IDS");
    if (raw == NULL || *raw == '\0') {
        raw = getenv("GPTQMODEL_VLLM_VISIBLE_PHYSICAL_IDS");
    } else {
        g_use_physical_span = 1;
    }
    if (raw == NULL || *raw == '\0') {
        return;
    }

    char *copy = strdup(raw);
    if (copy == NULL) {
        return;
    }

    char *cursor = copy;
    while (cursor != NULL && *cursor != '\0' && g_id_count < (sizeof(g_ids) / sizeof(g_ids[0]))) {
        char *next = strchr(cursor, ',');
        if (next != NULL) {
            *next = '\0';
        }

        char *end = NULL;
        long value = strtol(cursor, &end, 10);
        if (end != cursor && *end == '\0' && value >= 0 && value <= INT_MAX) {
            g_ids[g_id_count++] = (int)value;
        }

        cursor = next == NULL ? NULL : next + 1;
    }

    free(copy);
}

static unsigned int mapped_index(unsigned int index) {
    if (g_id_count == 0) {
        return index;
    }

    if (!g_use_physical_span) {
        if (index >= g_id_count) {
            return UINT_MAX;
        }
        return (unsigned int)g_ids[index];
    }

    for (size_t i = 0; i < g_id_count; ++i) {
        if ((unsigned int)g_ids[i] == index) {
            return index;
        }
        if ((unsigned int)g_ids[i] > index) {
            return (unsigned int)g_ids[i];
        }
    }

    return (unsigned int)g_ids[g_id_count - 1];
}

static void *resolve_symbol(const char *name) {
    void *symbol = dlsym(RTLD_NEXT, name);
    if (symbol != NULL) {
        return symbol;
    }

    if (g_nvml_handle == NULL) {
        g_nvml_handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    }
    if (g_nvml_handle == NULL) {
        return NULL;
    }
    return dlsym(g_nvml_handle, name);
}

static nvmlReturn_t call_real_count(const char *name, unsigned int *count) {
    nvml_count_fn fn = (nvml_count_fn)resolve_symbol(name);
    if (fn == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    return fn(count);
}

static nvmlReturn_t call_real_handle(const char *name, unsigned int index, nvmlDevice_t *device) {
    nvml_handle_fn fn = (nvml_handle_fn)resolve_symbol(name);
    if (fn == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    return fn(index, device);
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *count) {
    if (count == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    ensure_visible_ids();
    if (g_id_count > 0) {
        if (g_use_physical_span) {
            *count = (unsigned int)g_ids[g_id_count - 1] + 1U;
        } else {
            *count = (unsigned int)g_id_count;
        }
        return NVML_SUCCESS;
    }

    return call_real_count("nvmlDeviceGetCount_v2", count);
}

nvmlReturn_t nvmlDeviceGetCount(unsigned int *count) {
    if (count == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    ensure_visible_ids();
    if (g_id_count > 0) {
        if (g_use_physical_span) {
            *count = (unsigned int)g_ids[g_id_count - 1] + 1U;
        } else {
            *count = (unsigned int)g_id_count;
        }
        return NVML_SUCCESS;
    }

    return call_real_count("nvmlDeviceGetCount", count);
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device) {
    if (device == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    ensure_visible_ids();
    if (g_id_count > 0) {
        index = mapped_index(index);
        if (index == UINT_MAX) {
            return NVML_ERROR_INVALID_ARGUMENT;
        }
    }

    return call_real_handle("nvmlDeviceGetHandleByIndex_v2", index, device);
}

nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device) {
    if (device == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    ensure_visible_ids();
    if (g_id_count > 0) {
        index = mapped_index(index);
        if (index == UINT_MAX) {
            return NVML_ERROR_INVALID_ARGUMENT;
        }
    }

    return call_real_handle("nvmlDeviceGetHandleByIndex", index, device);
}
