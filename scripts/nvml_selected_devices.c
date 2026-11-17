/* Process-local NVML enumeration for a node containing an excluded failed GPU.
 * NCCL 2.21 enumerates all NVML devices before honoring CUDA visibility.
 * Intercept only its dlsym-resolved enumeration functions. All device handles,
 * PCI identities, capability queries and communication stay real.
 * Build: gcc -shared -fPIC -O2 -Wall -Wextra -o cache/nvml_selected_devices.so
 *        scripts/nvml_selected_devices.c -ldl -pthread
 * Activate only for the training Python process, not resource-query commands.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef int (*count_fn)(unsigned int *);
typedef int (*handle_fn)(unsigned int, void **);
typedef int (*index_fn)(void *, unsigned int *);
static unsigned int selected[32], selected_count;
static pthread_once_t selection_once = PTHREAD_ONCE_INIT;
static count_fn count_v1, count_v2;
static handle_fn handle_v1, handle_v2;
static index_fn device_index;

static void read_selection(void) {
    char *copy = strdup(getenv("ALFWORLD_NVML_PHYSICAL_INDICES"));
    char *save, *part = strtok_r(copy, ",", &save);
    while (part && selected_count < 32) {
        selected[selected_count++] = (unsigned int)strtoul(part, NULL, 10);
        part = strtok_r(NULL, ",", &save);
    }
    free(copy);
}

static int count_selected(count_fn fn, unsigned int *out) {
    unsigned int real_count;
    int rc = fn(&real_count);
    if (rc) return rc;
    pthread_once(&selection_once, read_selection);
    for (unsigned int i = 0; i < selected_count; ++i)
        if (selected[i] >= real_count) return 2; /* NVML_ERROR_INVALID_ARGUMENT */
    *out = selected_count;
    return 0;
}
static int get_count_v1(unsigned int *out) { return count_selected(count_v1, out); }
static int get_count_v2(unsigned int *out) { return count_selected(count_v2, out); }
static int handle_selected(handle_fn fn, unsigned int i, void **out) {
    pthread_once(&selection_once, read_selection);
    return i < selected_count ? fn(selected[i], out) : 2;
}
static int get_handle_v1(unsigned int i, void **out) { return handle_selected(handle_v1, i, out); }
static int get_handle_v2(unsigned int i, void **out) { return handle_selected(handle_v2, i, out); }
static int get_index(void *device, unsigned int *out) {
    unsigned int physical;
    int rc = device_index(device, &physical);
    if (rc) return rc;
    pthread_once(&selection_once, read_selection);
    for (unsigned int i = 0; i < selected_count; ++i)
        if (selected[i] == physical) { *out = i; return 0; }
    return 6; /* NVML_ERROR_NOT_FOUND: device was not selected */
}

void *dlsym(void *library, const char *name) {
    void *(*real_dlsym)(void *, const char *) = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    void *real = real_dlsym(library, name);
    if (!real || !getenv("ALFWORLD_NVML_PHYSICAL_INDICES")) return real;
    /* Legacy NVML v1 enumeration initializes every preceding card and fails
     * even for healthy indices after the lost GPU. v2 accesses them directly. */
    if (!strcmp(name, "nvmlInit")) return real_dlsym(library, "nvmlInit_v2");
    if (!strcmp(name, "nvmlDeviceGetCount")) { count_v1 = real_dlsym(library, "nvmlDeviceGetCount_v2"); return get_count_v1; }
    if (!strcmp(name, "nvmlDeviceGetCount_v2")) { count_v2 = real; return get_count_v2; }
    if (!strcmp(name, "nvmlDeviceGetHandleByIndex")) { handle_v1 = real_dlsym(library, "nvmlDeviceGetHandleByIndex_v2"); return get_handle_v1; }
    if (!strcmp(name, "nvmlDeviceGetHandleByIndex_v2")) { handle_v2 = real; return get_handle_v2; }
    if (!strcmp(name, "nvmlDeviceGetIndex")) { device_index = real; return get_index; }
    return real;
}
