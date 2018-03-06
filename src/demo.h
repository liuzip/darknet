#ifndef DEMO
#define DEMO

#include "image.h"
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, char *out_filename, int http_stream_port, int dont_show);

struct face_list_st{
    image face_feature;

    struct face_list_st *next;
};

#endif
