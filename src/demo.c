#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#ifdef WIN32
#include <time.h>
#include <winsock.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif

#include <unistd.h>

#define FRAMES 3

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/version.hpp"
#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio_c.h"
#endif
image get_image_from_stream(CvCapture *cap);
image crop_image_with_box(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes);
float compare_image(image im_1, image im_2);

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network net;
static image in   ;
static image in_s ;
static image det  ;
static image det_s;
static image disp = {0};
static image croped_im = {0};
static CvCapture * cap;
static float fps = 0;
static float demo_thresh = 0;

static float *predictions[FRAMES];
static int demo_index = 0;
static image images[FRAMES];
static IplImage* ipl_images[FRAMES];
static float *avg;

struct face_list_st *g_face_list = NULL;

void draw_detections_cv(IplImage* show_img, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes);
void show_image_cv_ipl(IplImage *disp, const char *name);
image get_image_from_stream_resize(CvCapture *cap, int w, int h, IplImage** in_img);
IplImage* in_img;
IplImage* det_img;
IplImage* show_img;

static int flag_exit;

void *fetch_in_thread(void *ptr)
{
    //in = get_image_from_stream(cap);
	in = get_image_from_stream_resize(cap, net.w, net.h, &in_img);
    if(!in.data){
        //error("Stream closed.");
		flag_exit = 1;
		return 0;
    }
    //in_s = resize_image(in, net.w, net.h);
	in_s = make_image(in.w, in.h, in.c);
	memcpy(in_s.data, in.data, in.h*in.w*in.c*sizeof(float));
	
    return 0;
}

void *detect_in_thread(void *ptr)
{
    float nms = .4;

    layer l = net.layers[net.n-1];
    float *X = det_s.data;
    float *prediction = network_predict(net, X);

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, FRAMES, l.outputs, avg);
    l.output = avg;

    free_image(det_s);
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, 0, 0);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");


    images[demo_index] = det;
    det = images[(demo_index + FRAMES/2 + 1)%FRAMES];
	ipl_images[demo_index] = det_img;
	det_img = ipl_images[(demo_index + FRAMES / 2 + 1) % FRAMES];
    demo_index = (demo_index + 1)%FRAMES;
	    
	//draw_detections(det, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
	draw_detections_cv(det_img, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);

#if 0
    // need siamese network
        layer feature_layer = net.layers[net.n-2];

        if(g_face_list == NULL){
            g_face_list = malloc(sizeof(struct face_list_st));
            memset(g_face_list, 0x0, sizeof(struct face_list_st));
            g_face_list->face_feature = get_convolutional_image(feature_layer);
            crop_image_with_box(det, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
        }
        else{
            image feature_map = get_convolutional_image(feature_layer);
            float threshold = 0.3;
            int existed = 0;

            struct face_list_st *face_ptr = NULL,
                *last_face = g_face_list;

            face_ptr = g_face_list;
            while(face_ptr){
                if(compare_image(face_ptr->face_feature, feature_map) < threshold){
                    existed = 1;
                    break;
                }

                last_face = face_ptr;
                face_ptr = face_ptr->next;
            }

            if(existed == 0){
                face_ptr = malloc(sizeof(struct face_list_st));
                memset(face_ptr, 0x0, sizeof(struct face_list_st));
                face_ptr->face_feature = feature_map;
                last_face->next = face_ptr;
                crop_image_with_box(det, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
            }
        }
#else
    crop_image_with_box(det, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
#endif


	return 0;
}

double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, 
	int frame_skip, char *prefix, char *out_filename, int http_stream_port, int dont_show)
{
    //skip = frame_skip;
    image **alphabet = load_alphabet();
    int delay = frame_skip;
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    printf("Demo\n");
    net = parse_network_cfg_custom(cfgfile, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);

    srand(2222222);

    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(!cap) error("Couldn't connect to webcam.\n");

    layer l = net.layers[net.n-1];
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) images[j] = make_image(1,1,3);

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

    pthread_t fetch_thread;
    pthread_t detect_thread;

    fetch_in_thread(0);
	det_img = in_img;
    det = in;
    det_s = in_s;

    fetch_in_thread(0);
    detect_in_thread(0);
    disp = det;
	det_img = in_img;
    det = in;
    det_s = in_s;

    for(j = 0; j < FRAMES/2; ++j){
        fetch_in_thread(0);
        detect_in_thread(0);
        disp = det;
		det_img = in_img;
        det = in;
        det_s = in_s;
    }

    int count = 0;
    if(!prefix && !dont_show){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        cvMoveWindow("Demo", 0, 0);
        cvResizeWindow("Demo", 1352, 1013);
    }

	CvVideoWriter* output_video_writer = NULL;    // cv::VideoWriter output_video;
	if (out_filename)
	{
		CvSize size;
		size.width = det_img->width, size.height = det_img->height;

		//const char* output_name = "test_dnn_out.avi";
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('H', '2', '6', '4'), 25, size, 1);
		output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('D', 'I', 'V', 'X'), 25, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('M', 'J', 'P', 'G'), 25, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('M', 'P', '4', 'V'), 25, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('M', 'P', '4', '2'), 25, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('X', 'V', 'I', 'D'), 25, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('W', 'M', 'V', '2'), 25, size, 1);
	}
	flag_exit = 0;

    double before = get_wall_time();

    while(1){
        ++count;
        if(1){
            if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
            if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

            if(!prefix){
				if (!dont_show) {
					show_image_cv_ipl(show_img, "Demo");
					int c = cvWaitKey(1);
					if (c == 10) {
						if (frame_skip == 0) frame_skip = 60;
						else if (frame_skip == 4) frame_skip = 0;
						else if (frame_skip == 60) frame_skip = 4;
						else frame_skip = 0;
					}
				}
            }else{
                char buff[256];
                sprintf(buff, "%s_%08d", prefix, count);
                save_image(disp, buff);
            }

			// if you run it with param -http_port 8090  then open URL in your web-browser: http://localhost:8090
			if (http_stream_port > 0 && show_img) {
				//int port = 8090;
				int port = http_stream_port;
				int timeout = 200;
				int jpeg_quality = 30;	// 1 - 100
				send_mjpeg(show_img, port, timeout, jpeg_quality);
			}

			// save video file
			if (output_video_writer && show_img) {
				cvWriteFrame(output_video_writer, show_img);
				printf("\n cvWriteFrame \n");
			}

			cvReleaseImage(&show_img);

            pthread_join(fetch_thread, 0);
            pthread_join(detect_thread, 0);

			if (flag_exit == 1) break;

            if(delay == 0){
                free_image(disp);
                disp  = det;
				show_img = det_img;
            }
			det_img = in_img;
            det   = in;
            det_s = in_s;
        }else {
            fetch_in_thread(0);
			det_img = in_img;
            det   = in;
            det_s = in_s;
            detect_in_thread(0);
            if(delay == 0) {
                free_image(disp);
                disp = det;
            }
			if (!dont_show) {
				show_image(disp, "Demo");
				cvWaitKey(1);
			}
        }
        --delay;
        if(delay < 0){
            delay = frame_skip;

            double after = get_wall_time();
            float curr = 1./(after - before);
            fps = curr;
            before = after;
        }
    }
	printf("input video stream closed. \n");
	if (output_video_writer) {
		cvReleaseVideoWriter(&output_video_writer);
		printf("output_video_writer closed. \n");
	}
}


#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, char *out_filename, int http_stream_port, int dont_show)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}

#endif


image crop_image_with_box(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes){
    int i;
    char buf[256] = {0x0};
    image tmp_im;
    time_t t;

    for(i = 0; i < num; ++i){
        int class_id = max_index(probs[i], classes);
        float prob = probs[i][class_id];
        if(prob > thresh && strcmp(names[class_id], "face") == 0){
            int width = im.h * .012;

            if(0){
                width = pow(prob, 1./2.)*10+1;
                alphabet = 0;
            }

            int offset = class_id*123457 % classes;
            float red = get_color(2,offset,classes);
            float green = get_color(1,offset,classes);
            float blue = get_color(0,offset,classes);
            float rgb[3];

            //width = prob*20+2;

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b = boxes[i];

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;
            /*
            fprintf(stderr, "[%s][%d] i[%d] x1[%d] y1[%d] x2[%d] y2[%d] width[%d]\n", __func__, __LINE__, 
                i, left, top, right, bot, width);
            */
            t = time(NULL);

            sprintf(buf, "%ld", time(&t));

            tmp_im = crop_image(im, left, top, abs(right - left), abs(bot - top));
            save_image(tmp_im, buf);
            free_image(tmp_im);
        }
    }
}

float compare_image(image im_1, image im_2){
    if(im_1.w != im_2.w ||
        im_1.h != im_2.h ||
        im_1.c != im_2.c){
        return 99999.9;
    }

    int width = im_1.w,
        height = im_1.h,
        channel = im_1.c,
        x, y, c;

    float norm = 0.0,
        diff = 0.0;

    for(x = 0; x < width; x ++){
        for(y = 0; y < height; y ++){
            for(c = 0; c < channel; c ++){
                diff = get_pixel(im_1, x, y, c) - get_pixel(im_2, x, y, c);
                norm = diff * diff;
            }
        }
    }

    return norm;
}

