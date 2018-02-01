
struct gt_handle;

#ifdef __cplusplus
extern "C" {
#else
typedef struct gt_handle gt_handle;
#endif

void gt_release(gt_handle *);
void gt_pull_bool(gt_handle *, char *, bool *, int, int *, int *);
void gt_pull_double(gt_handle *, char *, double *, int, int *, int *);
void gt_pull_float(gt_handle *, char *, float *, int, int *, int *);
void gt_pull_int(gt_handle *, char *, int *, int, int *, int *);
void gt_push_bool(gt_handle *, char *, bool *, int, int *, int *, bool);
void gt_push_double(gt_handle *, char *, double *, int, int *, int *, bool);
void gt_push_float(gt_handle *, char *, float *, int, int *, int *, bool);
void gt_push_int(gt_handle *, char *, int *, int, int *, int *, bool);
void gt_run(gt_handle *);

#ifdef __cplusplus
}
#endif
