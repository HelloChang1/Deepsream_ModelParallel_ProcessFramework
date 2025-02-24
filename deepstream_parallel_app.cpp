/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>

#include "nvds_yml_parser.h"
#include "gst-nvmessage.h"
#include "gstnvdsmeta.h"
#include "nvdsgstutils.h"
#include "nvbufsurface.h"
#include "nvdsmeta_schema.h"
#include "gstnvdsinfer.h"
#include "deepstream_common.h"
//cuda
#include "cuda.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cudla.h>

//OPENCV
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

// nvbufsurface
#include "nvbufsurftransform.h"

GST_DEBUG_CATEGORY (NVDS_APP);

#define MAX_DISPLAY_LEN 128

/*
people
car
bus
motorcycle
lamp
truck
*/
//用于输出每帧中不同类别物体的数量的条件判断宏定义
#define PGIE_CLASS_ID_PEOPLE 0
#define PGIE_CLASS_ID_CAR 1
#define PGIE_CLASS_ID_BUS 2
#define PGIE_CLASS_ID_MOTORCYCLE 3
#define PGIE_CLASS_ID_LAMP 4
#define PGIE_CLASS_ID_TRUCK 5


/* By default, OSD process-mode is set to CPU_MODE. To change mode, set as:
 * 1: GPU mode (for Tesla only)
 * 2: HW mode (For Jetson only)
 */
#define OSD_PROCESS_MODE 1

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 1

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
 /*如果输入流的分辨率不同，则必须设置多路复用器输出分辨率。复用器将把所有输入帧缩放到该分辨率*/
#define MUXER_OUTPUT_WIDTH 1024
#define MUXER_OUTPUT_HEIGHT 768

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

#define TILED_OUTPUT_WIDTH 2048
#define TILED_OUTPUT_HEIGHT 1536

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
 /*NVIDIA解码器源焊盘内存功能。此功能表示具有此功能的源焊盘将推送包含cuda缓冲区的GstBuffers*/
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

/* Check for parsing error. */
#define RETURN_ON_PARSER_ERROR(parse_expr) \
  if (NVDS_YAML_PARSER_SUCCESS != parse_expr) { \
    g_printerr("Error in parsing configuration file.\n"); \
    return -1; \
  }
#define CHECK_CUDA_STATUS(cuda_status,error_str) do { \
  if ((cuda_status) != cudaSuccess) { \
    g_print ("Error: %s in %s at line %d (%s)\n", \
        error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
  } \
} while (0)
// people
// car
// bus
// motorcycle
// lamp
// truck
//推理目标分类名
// gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
//   "RoadSign"
// };
gchar pgie_classes_str[6][32] = { "people", "car", "bus",
  "motorcycle","lamp","truck"
};
static gboolean PERF_MODE = FALSE;

  
static GstPadProbeReturn
nvvidconv_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  //申请一片内存并指向 osd_sink pad收到的数据
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0; 
    guint people_count = 0;
    guint car_count = 0;
    guint bus_count = 0;
    guint motorcycle_count = 0;
    guint lamp_count = 0;
    guint truck_count = 0;
    int offset = 0;
    int frame_count=0;


    // float obj_meta1_left,obj_meta1_top,obj_meta1_width,obj_meta1_height=0;
    // float obj_meta2_left,obj_meta2_top,obj_meta2_width,obj_meta2s_height=0;
   float obj1_people_position[4] = {0.0, 0.0, 0.0, 0.0};
   float obj1_car_position[4] = {0.0, 0.0, 0.0, 0.0};
   int display_length=0;
   char display_car_text[] = "";  
    //视频帧指针
    NvDsMetaList * l_frame = NULL;

    //物体指针
    NvDsMetaList * l_obj1 = NULL;
    NvDsMetaList * l_obj2 = NULL;

    NvDsMetaList * l_display1 = NULL;
    NvDsMetaList * l_display2 = NULL;

    NvDsDisplayMeta *display_meta1 = NULL;
    NvDsDisplayMeta *display_meta2 = NULL;
    NvDsObjectMeta *obj_meta1=NULL;
    NvDsObjectMeta *obj_meta2=NULL;

    NvDsFrameMeta *frame_meta=NULL;
    NvDsFrameMeta *frame_meta1=NULL;
    NvDsFrameMeta *frame_meta2=NULL;
    //获取一批元数据
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
      g_print ("l_frame_nums in batch:%d\n",batch_meta->num_frames_in_batch);

      for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
          frame_count++;
      }

      g_print ("l_frame_count:%d\n",frame_count);

    l_frame = batch_meta->frame_meta_list;
    frame_meta1 = (NvDsFrameMeta *) (l_frame->data);
    
    if (l_frame != NULL && l_frame->next != NULL) {  
    l_frame = l_frame->next;
    frame_meta2 = (NvDsFrameMeta *) (l_frame->data);
    }
    else
    {
       g_print ("l_frame->next: NULL!\n");
    }
    //----------------------------------------------------筛选source_id为0的视频流（微光视频流）-------------------------------------------
    if((frame_meta1->source_id) ==0)
    {
      num_rects=0;
      people_count=0;
      car_count=0;
      bus_count=0;
      motorcycle_count=0;
      lamp_count=0;
      truck_count=0;

      for (l_obj1 = frame_meta1->obj_meta_list; l_obj1 != NULL;
         l_obj1 = l_obj1->next) {
          obj_meta1 = (NvDsObjectMeta *) (l_obj1->data);
            //根据物体元数据的分类id，进行计数，统计不同类别物体的个数
            /*people car bus motorcycle lamp truck*/
            // NvDsComp_BboxInfo *single_obj=obj_meta1->detector_bbox_info;
            if (obj_meta1->class_id == PGIE_CLASS_ID_PEOPLE) {
              obj_meta1->rect_params.border_color.red=1;
              obj_meta1->rect_params.border_color.green=0;
              obj_meta1->rect_params.border_color.blue=0;
              obj_meta1->rect_params.border_color.alpha=1;
              obj1_people_position[0] = obj_meta1->rect_params.left;   // 左边界  
              obj1_people_position[1] = obj_meta1->rect_params.top;    // 上边界  
              obj1_people_position[2] = obj_meta1->rect_params.width;  // 宽
              obj1_people_position[3] = obj_meta1->rect_params.height; // 高
              people_count++;
              num_rects++;
            }
            if (obj_meta1->class_id == PGIE_CLASS_ID_CAR) {
              obj_meta1->rect_params.border_color.red=0;
              obj_meta1->rect_params.border_color.green=1;
              obj_meta1->rect_params.border_color.blue=0;
              obj_meta1->rect_params.border_color.alpha=1;
              // snprintf(obj_meta1->text_params.display_text, 128, "%d%s%.2f", obj_meta1->class_id, obj_meta1->obj_label[obj_meta1->class_id], obj_meta1->confidence);  
              // obj_meta1->text_params.display_text="Pelople:1";
              obj_meta1->text_params.set_bg_clr=1;
              obj_meta1->text_params.text_bg_clr.red=0.5;
              obj_meta1->text_params.text_bg_clr.green=0;
              obj_meta1->text_params.text_bg_clr.blue=0.5;
              obj_meta1->text_params.text_bg_clr.alpha=1;
              obj1_car_position[0] = obj_meta1->rect_params.left;   // 左边界  
              obj1_car_position[1] = obj_meta1->rect_params.top;    // 上边界  
              obj1_car_position[2] = obj_meta1->rect_params.width;  // 宽
              obj1_car_position[3] = obj_meta1->rect_params.height; // 高
                car_count++;
                num_rects++;
            }
            if (obj_meta1->class_id == PGIE_CLASS_ID_BUS) {
              obj_meta1->rect_params.border_color.red=0;
              obj_meta1->rect_params.border_color.green=0;
              obj_meta1->rect_params.border_color.blue=1;
              obj_meta1->rect_params.border_color.alpha=1;
                bus_count++;
                num_rects++;
            }
            if (obj_meta1->class_id == PGIE_CLASS_ID_MOTORCYCLE) {
              obj_meta1->rect_params.border_color.red=0.5;
              obj_meta1->rect_params.border_color.green=0.5;
              obj_meta1->rect_params.border_color.blue=0.5;
              obj_meta1->rect_params.border_color.alpha=1;
                motorcycle_count++;
                num_rects++;
            }
            if (obj_meta1->class_id == PGIE_CLASS_ID_LAMP) {
              obj_meta1->rect_params.border_color.red=0.1;
              obj_meta1->rect_params.border_color.green=0.5;
              obj_meta1->rect_params.border_color.blue=0.5;
              obj_meta1->rect_params.border_color.alpha=1;
                lamp_count++;
                num_rects++;
            }
            if (obj_meta1->class_id == PGIE_CLASS_ID_TRUCK) {
              obj_meta1->rect_params.border_color.red=0.5;
              obj_meta1->rect_params.border_color.green=0.1;
              obj_meta1->rect_params.border_color.blue=0.5;
              obj_meta1->rect_params.border_color.alpha=1;
                truck_count++;
                num_rects++;
            }
         }
      g_print ("Frame source_id = %d Frame Number = %d Number of objects = %d "
            "people Count = %d car Count = %d bus Count = %d motorcycle Count = %d lamp Count = %d truck Count = %d\n",
            frame_meta1->source_id,frame_meta1->frame_num, num_rects, people_count, car_count,bus_count,motorcycle_count,lamp_count,truck_count);      
    }
    //----------------------------------------------------筛选source_id为1的视频流（红外视频流）-------------------------------------------
    if((frame_meta2->source_id) ==1)
    {
      nvds_copy_obj_meta_list(frame_meta1->obj_meta_list,frame_meta2);            
    }
    return GST_PAD_PROBE_OK;
}
//融合模型的探针函数
static GstPadProbeReturn
ronghe_tiler_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;

  guint num_rects = 0; 
  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList * l_frame = NULL;
  NvDsMetaList * l_obj = NULL;
  NvDsDisplayMeta *display_meta = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

#if 1
  // Get original raw data
  GstMapInfo in_map_info;
  if (!gst_buffer_map (buf, &in_map_info, GST_MAP_READ)) {
      g_print ("Error: Failed to map gst buffer\n");
      gst_buffer_unmap (buf, &in_map_info);
      return GST_PAD_PROBE_OK;
  }

  NvBufSurface *surface = (NvBufSurface *)in_map_info.data;
#endif

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
    l_frame = l_frame->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
      for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
              l_obj = l_obj->next) {
          obj_meta = (NvDsObjectMeta *) (l_obj->data);
          // if (obj_meta->class_id % 3 == 0) {
          //   obj_meta->rect_params.border_color.red = 1.0;
          //   obj_meta->rect_params.border_color.green = 0.0;
          //   obj_meta->rect_params.border_color.blue = 0.0;
          //   obj_meta->rect_params.border_color.alpha = 0.5;
          // }
          // if (obj_meta->class_id % 3 == 1) {
          //   obj_meta->rect_params.border_color.red = 0.0;
          //   obj_meta->rect_params.border_color.green = 1.0;
          //   obj_meta->rect_params.border_color.blue = 0.0;
          //   obj_meta->rect_params.border_color.alpha = 0.5;
          // }
          // if (obj_meta->class_id % 3 == 2) {
          //   obj_meta->rect_params.border_color.red = 0.0;
          //   obj_meta->rect_params.border_color.green = 0.0;
          //   obj_meta->rect_params.border_color.blue = 1.0;
          //   obj_meta->rect_params.border_color.alpha = 0.5;
          // }
          num_rects++;
      }
      g_print ("In the ronghe_tiler_src_pad_buffer_probe\n");
      g_print ("TILER: Frame Number = %d Number of objects = %d\n", frame_meta->frame_num, num_rects);

#if 0
      /* To verify  encoded metadata of cropped frames, we iterate through the
      * user metadata of each frame and if a metadata of the type
      * 'NVDS_CROP_IMAGE_META' is found then we write that to a file as
      * implemented below.
      */
      char fileFrameNameString[FILE_NAME_SIZE];
      const char *osd_string = "tiler";
      /* For Demonstration Purposes we are writing metadata to jpeg images of
        * the first 10 frames only.
        * The files generated have an 'OSD' prefix. */
      if (frame_number < 11) {
        NvDsUserMetaList *usrMetaList = frame_meta->frame_user_meta_list;
        FILE *file;
        int stream_num = 0;
        while (usrMetaList != NULL) {
          NvDsUserMeta *usrMetaData = (NvDsUserMeta *) usrMetaList->data;
          if (usrMetaData->base_meta.meta_type == NVDS_CROP_IMAGE_META) {
            snprintf (fileFrameNameString, FILE_NAME_SIZE, "%s_frame_%d_%d.jpg",
                osd_string, frame_number, stream_num++);
            NvDsObjEncOutParams *enc_jpeg_image =
                (NvDsObjEncOutParams *) usrMetaData->user_meta_data;
            /* Write to File */
            file = fopen (fileFrameNameString, "wb");
            fwrite (enc_jpeg_image->outBuffer, sizeof (uint8_t),
                enc_jpeg_image->outLen, file);
            fclose (file);
          }
          else if(usrMetaData->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META){
            NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) usrMetaData->user_meta_data;
            for (unsigned int i = 0; i < meta->num_output_layers; i++) {
              NvDsInferLayerInfo *info = &meta->output_layers_info[i];
              info->buffer = meta->out_buf_ptrs_host[i];
              if (meta->out_buf_ptrs_dev[i]) {
                cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                    info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
              }
            }
            size_t ch = meta->output_layers_info->inferDims.d[0];
            size_t height = meta->output_layers_info->inferDims.d[1];
            size_t width = meta->output_layers_info->inferDims.d[2];
            size_t o_count = meta->output_layers_info->inferDims.numElements;
            // cvcore::Image<cvcore::ImageType::RGB_F32> img(width, height, width * sizeof(float), (float *) meta->output_layers_info[0].buffer, TRUE);
            float *outputCoverageBuffer =(float *) meta->output_layers_info[0].buffer;
            uint8_t* uint8Buffer = (uint8_t *)malloc(o_count*sizeof(uint8_t));

            for(int o_index=0; o_index < o_count; o_index++){
              // outputCoverageBuffer[o_index] *= 255.0f;
              uint8Buffer[o_index] = static_cast<uint8_t>(std::min(std::max(outputCoverageBuffer[o_index] * 255.0f, 0.0f), 255.0f));
            }
            NvDsObjEncOutParams *enc_jpeg_image = (NvDsObjEncOutParams *)malloc(sizeof(NvDsObjEncOutParams));
            enc_jpeg_image->outBuffer = uint8Buffer;
            enc_jpeg_image->outLen = o_count;
            snprintf (fileFrameNameString, FILE_NAME_SIZE, "%s_frame_%d_%d.jpg",
                  osd_string, frame_number, stream_num++);
            file = fopen (fileFrameNameString, "wb");
            fwrite (enc_jpeg_image->outBuffer, sizeof (uint8_t),
                  enc_jpeg_image->outLen, file);
            fclose (file);
          // g_print ("SIZE: %ld \n", sizeof(*outputCoverageBuffer));
            // std::vector < NvDsInferLayerInfo >
            // outputLayersInfo (meta->output_layers_info,
            // meta->output_layers_info + meta->num_output_layers);
          }
          usrMetaList = usrMetaList->next;
        }
      }
#endif

#if 1
    /**保存指向用于帧的@ref NvDsUserMeta类型指针列表的指针*/
    NvDsUserMetaList *usrMetaList = frame_meta->frame_user_meta_list;
    if (usrMetaList != NULL) 
    {
      NvDsUserMeta *usrMetaData = (NvDsUserMeta *) usrMetaList->data;

      if(usrMetaData->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META){
          // NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
          //TODO for cuda device memory we need to use cudamemcpy
          NvBufSurfaceMap (surface, -1, -1, NVBUF_MAP_READ);
          /* Cache the mapped data for CPU access */
          NvBufSurfaceSyncForCpu (surface, 0, 0); //will do nothing for unified memory type on dGPU
          guint surface_height = surface->surfaceList[frame_meta->batch_id].height;
          guint surface_width = surface->surfaceList[frame_meta->batch_id].width;

          //Create Mat from NvMM memory, refer opencv API for how to create a Mat
          cv::Mat nv12_mat = cv::Mat(surface_height*3/2, surface_width, CV_8UC1, surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0],
          surface->surfaceList[frame_meta->batch_id].pitch);

          NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) usrMetaData->user_meta_data;
          for (unsigned int i = 0; i < meta->num_output_layers; i++) {
            NvDsInferLayerInfo *info = &meta->output_layers_info[i];
            info->buffer = meta->out_buf_ptrs_host[i];
            if (meta->out_buf_ptrs_dev[i]) {
              cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                  info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
            }
          }

          //Create image from NVDSINFER_TENSOR_OUTPUT_META
          guint ch = meta->output_layers_info->inferDims.d[0];
          guint fusion_height = meta->output_layers_info->inferDims.d[1];
          guint fusion_width = meta->output_layers_info->inferDims.d[2];
          guint o_count = meta->output_layers_info->inferDims.numElements;
          guint onechannel_size = fusion_height * fusion_width;
          float *outputCoverageBuffer =(float *) meta->output_layers_info[0].buffer;

          cv::Mat fusion_mat; 
          using image_type = uint8_t;
          int image_format = CV_8UC1;
          image_type* uint8Buffer = (image_type *)malloc(o_count * sizeof(image_type));
          
          for(int o_index=0; o_index < o_count; o_index++){
            uint8Buffer[o_index] = static_cast<uint8_t>(std::min(std::max(outputCoverageBuffer[o_index] * 255.0f, 0.0f), 255.0f));
          }

          fusion_mat = cv::Mat(fusion_height, fusion_width, image_format, uint8Buffer, fusion_width);
          
            NvBufSurface *inter_buf = nullptr;
            NvBufSurfaceCreateParams create_params;
            create_params.gpuId  = surface->gpuId;
            create_params.width  = surface_width;
            create_params.height = surface_height;
            create_params.colorFormat = NVBUF_COLOR_FORMAT_GRAY8;
            create_params.layout = NVBUF_LAYOUT_PITCH;
          #ifdef __aarch64__
            create_params.memType = NVBUF_MEM_DEFAULT;
          #else
            create_params.memType = NVBUF_MEM_CUDA_UNIFIED;
          #endif

            //Create another scratch RGBA NvBufSurface
            if (NvBufSurfaceCreate (&inter_buf, 1,
              &create_params) != 0) {
              GST_ERROR ("Error: Could not allocate internal buffer ");
              return GST_PAD_PROBE_OK;
            }
            if(NvBufSurfaceMap (inter_buf, 0, -1, NVBUF_MAP_READ_WRITE) != 0)
              std::cout << "map error" << std::endl;
            NvBufSurfaceSyncForCpu (inter_buf, -1, -1);
            cv::Mat trans_mat = cv::Mat(surface_height, surface_width, CV_8UC1, inter_buf->surfaceList[frame_meta->batch_id].mappedAddr.addr[0],
          inter_buf->surfaceList[0].pitch);
          
          cv::Mat gray;
          // temp.copyTo(trans_mat);

          // cv::Mat dstROI = trans_mat(cv::Rect(0, 0, fusion_mat.cols, fusion_mat.rows));
          cv::cvtColor(nv12_mat, gray, cv::COLOR_YUV2GRAY_NV12);
          // 将源矩阵复制到目标矩阵的ROI区域
          cv::Mat dstROI = gray(cv::Rect(0, fusion_height, fusion_width, fusion_height));
          
          fusion_mat.copyTo(dstROI);

          gray.copyTo(trans_mat);

          NvBufSurfaceSyncForDevice(inter_buf, -1, -1);
          inter_buf->numFilled = 1;
          NvBufSurfTransformConfigParams transform_config_params;
          NvBufSurfTransformParams transform_params;
          NvBufSurfTransformRect src_rect;
          NvBufSurfTransformRect dst_rect;
          cudaStream_t cuda_stream;
          CHECK_CUDA_STATUS (cudaStreamCreate (&cuda_stream),
            "Could not create cuda stream");
          // transform_config_params.input_buf_count = 2;
          // transform_config_params.composite_flag = NVBUFSURF_TRANSFORM_COMPOSITE;

          transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;
          transform_config_params.gpu_id = surface->gpuId;
          transform_config_params.cuda_stream = cuda_stream;
          /* Set the transform session parameters for the conversions executed in this
            * thread. */
          NvBufSurfTransform_Error err = NvBufSurfTransformSetSessionParams (&transform_config_params);
          if (err != NvBufSurfTransformError_Success) {
            std::cout <<"NvBufSurfTransformSetSessionParams failed with error "<< err << std::endl;
            return GST_PAD_PROBE_OK;
          }
          /* Set the transform ROIs for source and destination, only do the color format conversion*/
          src_rect = {0, 0, surface_width, surface_height };
          dst_rect = {0, 0, surface_width, surface_height};

          /* Set the transform parameters */
          transform_params.src_rect = &src_rect;
          transform_params.dst_rect = &dst_rect;
          transform_params.transform_flag =
            NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC |
              NVBUFSURF_TRANSFORM_CROP_DST;
          transform_params.transform_filter = NvBufSurfTransformInter_Default;

          /* Transformation format conversion, Transform rotated RGBA mat to NV12 memory in original input surface*/
          err = NvBufSurfTransform (inter_buf, surface, &transform_params);
          if (err != NvBufSurfTransformError_Success) {
            std::cout<< "NvBufSurfTransform failed with error %d while converting buffer" << err <<std::endl;
            return GST_PAD_PROBE_OK;
          }
          // nvds_copy_obj_meta();
          NvBufSurfaceUnMap(inter_buf, 0, 0);
        }
    }
    NvBufSurfaceUnMap(surface, 0, 0);
#endif 

  }
  // frame_number++;
  return GST_PAD_PROBE_OK;
}




#if 1
//总线应答，处理消息，分析错误
static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  // 判断总线上接收到的消息类别
  switch (GST_MESSAGE_TYPE (msg)) {
    //如果是EOS，则停止loop
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;

    case GST_MESSAGE_WARNING:
    {
      gchar *debug;
      GError *error;
      //获取错误信息并定位到元件
      gst_message_parse_warning (msg, &error, &debug);
      g_printerr ("WARNING from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      g_free (debug);
      g_printerr ("Warning: %s\n", error->message);
      g_error_free (error);
      break;
    }
    // case GST_STATE_READY:
    //   GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (appCtx->pipeline.
    //   pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "ds-app-ready");
    //   if (oldstate == GST_STATE_NULL) {
    //       NVGSTDS_INFO_MSG_V ("Pipeline ready\n");
    //     } else {
    //       NVGSTDS_INFO_MSG_V ("Pipeline stopped\n");
    //     }
    //   break;

    case GST_MESSAGE_ERROR:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }

    case GST_MESSAGE_ELEMENT:
    {
      if (gst_nvmessage_is_stream_eos (msg)) {
        guint stream_id;
        if (gst_nvmessage_parse_stream_eos (msg, &stream_id)) {
          g_print ("Got EOS from stream %d\n", stream_id);
        }
      }
      break;
    }
    default:
      break;
  }
  return TRUE;
}
#else

#endif

//用于urldecoderbin，用于创建新的pad
  // g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
  //     G_CALLBACK (cb_newpad), bin);
static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  /*
  尝试获取 decoder_src_pad 上当前的媒体类型（caps）。
  媒体类型（caps）是一个描述媒体数据格式的 GStreamer 结构，例如视频格式（如 YUV420P）、音频格式（如 PCM）等。
  */
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  //如果是无效的
  if (!caps) {
    // 查询 Pad 可能支持的所有媒体类型（caps）
    caps = gst_pad_query_caps (decoder_src_pad, NULL);
  }

  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);

  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    /*只有当decodebin选择了nvidia解码器插件nvdec_*时，
    才链接decodebin pad。我们通过检查焊盘盖是否包含NVMM内存功能来完成此操作*/
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");

      if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
              decoder_src_pad)) {
        g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref (bin_ghost_pad);
    } else {
      g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}
  // g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
  //     G_CALLBACK (decodebin_child_added), bin);
static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "decodebin") == name) {

    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }

  if (g_strrstr (name, "source") == name) {
        g_object_set(G_OBJECT(object),"drop-on-latency",true,NULL);
  }

}

//根据输入视频源的数量，创建多个source箱柜，在箱柜中增加了uri_decode_bin元件
static GstElement *
create_source_bin (guint index, gchar * uri)
{
  // 创建bin与uri_decode_bin元件
  GstElement *bin = NULL, 
  *uri_decode_bin = NULL;

  gchar bin_name[16] = { };
  g_snprintf (bin_name, 15, "source-bin-%02d", index);

  /* Create a source GstBin to abstract this bin's content from the rest of the pipeline */
  // 创建一个源GstBin，以从管道的其余部分中提取此bin的内容
  // 创建一个source 箱柜，箱柜名为"source-bin-01"，"source-bin-02"，"source-bin-03"，"source-bin-04"
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the stream and the codec and plug the appropriate demux and decode plugins. */
  // 我们将使用decodebin，让它计算出流和编解码器的容器格式，并插入适当的解复用器和解码插件。
  //PERF_MODE default false
  if (PERF_MODE) {
    uri_decode_bin = gst_element_factory_make ("nvurisrcbin", "uri-decode-bin");
    g_object_set (G_OBJECT (uri_decode_bin), "file-loop", TRUE, NULL);
    g_object_set (G_OBJECT (uri_decode_bin), "cudadec-memtype", 0, NULL);
  } 
  else
   {
    // gst_element_factory_make 此函数为新创建的元件采用工厂名称和元件名称。
    // uridecodebin：

    uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");
  }

  if (!bin || !uri_decode_bin) {
    g_printerr ("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  //将视频的url指向uri_decode_bin元件
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a callback once a new pad for raw data has beed created by the decodebin */
  // 连接到decodebin的“pad-added”信号，一旦解码器创建了新的原始数据焊盘，就会生成回调
  // uri_decode_bin自带demuxer，接收到数据之后它就有了足够的信息生成source pad。这时我们就可以继续把其他部分和demuxer新生成的pad连接起来

  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);
 
 //创建source箱柜，用于多个视频输入解码
  gst_bin_add (GST_BIN (bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  /*我们需要为源bin创建一个ghost pad，它将作为视频解码器src pad的代理。
  重影垫现在不会有目标。
  一旦解码箱创建了视频解码器并生成cb_newpad回调，我们将把重影焊盘目标设置为视频解码器rc焊盘*/
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",GST_PAD_SRC)))
              {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }
  //返回创建好的source箱柜，并且添加了精灵衬垫ghost_pad
  return bin;
}

gboolean
link_element_to_metamux_sink_pad (GstElement *metamux, GstElement *elem,
    gint index)
{
  gboolean ret = FALSE;
  GstPad *mux_sink_pad = NULL;
  GstPad *src_pad = NULL;
  gchar pad_name[16];

  if (index >= 0) {
    g_snprintf (pad_name, 16, "sink_%u", index);
    pad_name[15] = '\0';
  } else {
    strcpy (pad_name, "sink_%u");
  }

  mux_sink_pad = gst_element_get_request_pad (metamux, pad_name);
  if (!mux_sink_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get sink pad from metamux");
    goto done;
  }

  src_pad = gst_element_get_static_pad (elem, "src");
  if (!src_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get src pad from '%s'",
                        GST_ELEMENT_NAME (elem));
    goto done;
  }

  if (gst_pad_link (src_pad, mux_sink_pad) != GST_PAD_LINK_OK) {
    NVGSTDS_ERR_MSG_V ("Failed to link '%s' and '%s'", GST_ELEMENT_NAME (metamux),
        GST_ELEMENT_NAME (elem));
    goto done;
  }

  ret = TRUE;

done:
  if (mux_sink_pad) {
    gst_object_unref (mux_sink_pad);
  }
  if (src_pad) {
    gst_object_unref (src_pad);
  }
  return ret;
}

int
main (int argc, char *argv[])
{
  //定义GStreamer变量
  GMainLoop *loop = NULL;

  GstElement *pipeline = NULL, 
  *streammux = NULL, 
  *streamdemux = NULL,
  *tee_b0=NULL,
  *tee_b1=NULL, 
  *queue_b0=NULL,
  *queue_b1=NULL,
  *queue_b2=NULL,
  *queue_b3=NULL,
  *streammux_b0=NULL,
  *streammux_b1=NULL,
  *sink = NULL, 
  *pgie_b0 = NULL,
  *pgie_b1 = NULL,
  *metamuxer = NULL,
  *queue1, *queue2, *queue3, *queue4, *queue5, *nvvidconv = NULL,
  *nvosd = NULL, 
  *tiler = NULL, 
  *nvdslogger = NULL;
  GstBus *bus = NULL;

  guint bus_watch_id;
  GstPad *nvvidconv_sink_pad = NULL;
  guint i =0, num_sources = 0;
  guint tiler_rows, tiler_columns;
  guint pgie_batch_size;
  gboolean yaml_config = FALSE;
  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;

// 如果环境变量NVDS_TEST3_PERF_MODE存在且其值为"1"，则PERF_MODE将被设置为true；否则，PERF_MODE将被设置为false
  PERF_MODE = g_getenv("NVDS_TEST3_PERF_MODE") && !g_strcmp0(g_getenv("NVDS_TEST3_PERF_MODE"), "1");

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Check input arguments */
  if (argc < 2) {
    g_printerr ("Usage: %s <yml file>\n", argv[0]);
    g_printerr ("OR: %s <uri1> [uri2] ... [uriN] \n", argv[0]);
    return -1;
  }
	//GST_DEBUG_DUMP_DOT_DIR=/opt/nvidia/deepstream/deepstream-6.2/sources/apps/sample_apps/dotfile 
  /* Standard GStreamer initialization */
  // g_setenv("GST_DEBUG_DUMP_DOT_DIR", "/opt/nvidia/deepstream/deepstream-6.2/sources/apps/sample_apps/dotfile/", TRUE);
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Parse inference plugin type */
  yaml_config = (g_str_has_suffix (argv[1], ".yml") ||
          g_str_has_suffix (argv[1], ".yaml"));

  if (yaml_config) {
    RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1],
                "primary-gie-0"));
  }

  /* Create gstreamer elements */
/*---------------------------------------------------create Pipeline & streammux element-------------------------------- */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dstest3-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  //创建pipeline箱柜，并在其中添加streammux元件
  gst_bin_add (GST_BIN (pipeline), streammux);
/*---------------------------------------------------create Pipeline & streammux element-------------------------------- */

/*---------------------------------------------------Parse yaml or command for source_nums-------------------------------- */
// typedef struct _GList GList;

// struct _GList
// {
//   gpointer data;
//   GList *next;
//   GList *prev;
// };
  GList *src_list = NULL ;

  if (yaml_config) 
  {
    //获取dstest3_config.yml配置文件中source-list项的视频源列表，并存储在src_list结构体指针中
    RETURN_ON_PARSER_ERROR(nvds_parse_source_list(&src_list, argv[1], "source-list"));

    GList * temp = src_list;
    //循环遍历视频源，获取视频源数量
    while(temp) {
      num_sources++;
      temp=temp->next;
    }
    g_print ("num_sources:%d\n",num_sources);

    g_list_free(temp);
  } 
  else 
  {
    // 如果不是yaml文件，则通过命令行获取视频源数量
      num_sources = argc - 1;
  }
/*---------------------------------------------------Parse yaml or command for source_nums-------------------------------- */

/*---------------------------------------------------build sourcebin and connect it to streammux-------------------------------- */

  for (i = 0; i < num_sources; i++) {
    //声明sink(输入)，src（输出）衬垫
    GstPad *sinkpad, *srcpad;

    gchar pad_name[16] = { };

    //--------------根据输入视频源的数量，创建video_nums个sourcebin元件，作为pipeline箱柜的元件之一
    GstElement *source_bin= NULL;

    if (g_str_has_suffix (argv[1], ".yml") || g_str_has_suffix (argv[1], ".yaml")) 
    {
      g_print("Now playing : %s\n",(char*)(src_list)->data);

      source_bin = create_source_bin (i, (char*)(src_list)->data);
    } 
    else 
    {
      source_bin = create_source_bin (i, argv[i + 1]);
    }
    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }
    gst_bin_add (GST_BIN (pipeline), source_bin);
    //--------------根据输入视频源的数量，创建video_nums个sourcebin元件，作为pipeline箱柜的元件之一


    //Request_pad 根据需求建立，不能自动连接element
    g_snprintf (pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_get_request_pad (streammux, pad_name);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }
    // 获取元素上静态定义的 Pad
    srcpad = gst_element_get_static_pad (source_bin, "src");
    if (!srcpad) {
      g_printerr ("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref (srcpad);
    gst_object_unref (sinkpad);
    //下一个视频输入源
    if (yaml_config) {
      src_list = src_list->next;
    }
  }

  if (yaml_config) {
    g_list_free(src_list);
  }
/*---------------------------------------------------build sourcebin and connect it to streammux-------------------------------- */

/*---------------------------------------------------build the rest elements-------------------------------- */
  /* Use nvinfer or nvinferserver to infer on batched frame. */
  if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
    pgie_b0 = gst_element_factory_make ("nvinfer", "primary-nvinference-engine_b0");
    pgie_b1 = gst_element_factory_make ("nvinfer", "primary-nvinference-engine_b1");
  } else {
    pgie_b0 = gst_element_factory_make ("nvinfer", "primary-nvinference-engine_b0");
    pgie_b1 = gst_element_factory_make ("nvinfer", "primary-nvinference-engine_b1");

  }

  //创建nvstreamdemux 
  streamdemux =  gst_element_factory_make ("nvstreamdemux", "nvdemux");
  g_object_set (G_OBJECT (streamdemux), "per-stream-eos", TRUE, NULL);

  tee_b0 = gst_element_factory_make ("tee", "tee_b0");
  tee_b1 = gst_element_factory_make ("tee", "tee_b1");

  streammux_b0 = gst_element_factory_make ("nvstreammux", "stream-muxer_b0");
  streammux_b1 = gst_element_factory_make ("nvstreammux", "stream-muxer_b1");

  queue_b0 = gst_element_factory_make ("queue", "queue_b0");
  queue_b1 = gst_element_factory_make ("queue", "queue_b1");
  queue_b2 = gst_element_factory_make ("queue", "queue_b2");
  queue_b3 = gst_element_factory_make ("queue", "queue_b3");

  metamuxer = gst_element_factory_make ("nvdsmetamux", "infer_bin_muxer");

  /* Add queue elements between every two elements */
  /*
  在 GStreamer 中，"queue" 是一个元素，它用于在管道（pipeline）中缓冲媒体数据。
  这可以用于防止数据丢失，尤其是在不同元素之间的处理速度不匹配时。
  */
  queue1 = gst_element_factory_make ("queue", "queue1");
  queue2 = gst_element_factory_make ("queue", "queue2");
  queue3 = gst_element_factory_make ("queue", "queue3");
  queue4 = gst_element_factory_make ("queue", "queue4");
  queue5 = gst_element_factory_make ("queue", "queue5");

  /* Use nvdslogger for perf measurement. */
  nvdslogger = gst_element_factory_make ("nvdslogger", "nvdslogger");

  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  if (PERF_MODE) {
    sink = gst_element_factory_make ("fakesink", "nvvideo-renderer");
  } else {
    /* Finally render the osd output */
    if(prop.integrated) {
      sink = gst_element_factory_make ("nv3dsink", "nv3d-sink");
    } else {
      sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
    }
  }

  if (!pgie_b0 || !pgie_b1 || !streamdemux || !tee_b0 || !tee_b1 || !queue_b0 || !queue_b1 || !queue_b2 || !queue_b3 || !streammux_b0 || !streammux_b1 || !metamuxer || !nvdslogger || !tiler || !nvvidconv || !nvosd || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
/*---------------------------------------------------build the rest elements-------------------------------- */

/*---------------------------------------------------parse the config file-------------------------------- */
  if (yaml_config) {

    RETURN_ON_PARSER_ERROR(nvds_parse_streammux(streammux, argv[1],"streammux"));

    // RETURN_ON_PARSER_ERROR(nvds_parse_gie(metamuxer, argv[1], "primary-gie"));
    // g_object_set (G_OBJECT (bin->muxer), "config-file",
		//   GET_FILE_PATH (config->meta_mux_config.config_file_path), NULL);

/*------------------------------------------解析nvsinfer-b0 & nvsinfer-b1配置文件选项----------------------------------------*/ 
    RETURN_ON_PARSER_ERROR(nvds_parse_gie(pgie_b0, argv[1], "primary-gie-0"));
    RETURN_ON_PARSER_ERROR(nvds_parse_gie(pgie_b1, argv[1], "primary-gie-1"));

    g_object_get (G_OBJECT (pgie_b0), "batch-size", &pgie_batch_size, NULL);
    //如果推理的配置文件里batch_size不等于输入的视频源数量
    if (pgie_batch_size != num_sources) {
      g_printerr
          ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
          pgie_batch_size, num_sources);
          // 将批次大小设置为与视频源数量相同
      g_object_set (G_OBJECT (pgie_b0), "batch-size", num_sources, NULL);
    }

/*------------------------------------------解析nvsinfer-b0 & nvsinfer-b1配置文件选项----------------------------------------*/ 


/*------------------------------------------解析streammux-b0 & streammux-b1配置文件选项----------------------------------------*/ 
    RETURN_ON_PARSER_ERROR(nvds_parse_streammux(streammux_b0, argv[1],"streammux-b0"));
    RETURN_ON_PARSER_ERROR(nvds_parse_streammux(streammux_b1, argv[1],"streammux-b1"));

/*------------------------------------------解析streammux-b0 & streammux-b1配置文件选项----------------------------------------*/ 

/*------------------------------------------解析streammux-b0 & streammux-b1配置文件选项----------------------------------------*/ 

/*------------------------------------------设置metamuxer配置文件选项----------------------------------------*/ 
  const gchar *config_file_path  = "/workspace/chang/deepstream_docker/sources/apps/sample_apps/deepstream-test3/data/config_metamux3.txt";
  // g_object_set (G_OBJECT (metamuxer), "enable", 1 , NULL);
  g_object_set (G_OBJECT (metamuxer), "config-file",GET_FILE_PATH (config_file_path), NULL);
/*------------------------------------------设置umetamuxer配置文件选项----------------------------------------*/ 

/*------------------------------------------解析nvsosd配置文件选项----------------------------------------*/ 
    RETURN_ON_PARSER_ERROR(nvds_parse_osd(nvosd, argv[1],"osd"));
/*------------------------------------------解析nvsosd配置文件选项----------------------------------------*/ 

/*------------------------------------------解析nvtiler配置文件选项----------------------------------------*/ 
    //设置显示的布局
    // tiler_rows = (guint) sqrt (num_sources);
    // tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
    // g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns, NULL);
    RETURN_ON_PARSER_ERROR(nvds_parse_tiler(tiler, argv[1], "tiler"));
/*------------------------------------------解析nvtiler配置文件选项----------------------------------------*/ 

/*------------------------------------------解析nvssink配置文件选项----------------------------------------*/     
    if(prop.integrated) {
      if(PERF_MODE==1)
      {
      RETURN_ON_PARSER_ERROR(nvds_parse_fake_sink(sink, argv[1], "sink"));
      }
    else {
      RETURN_ON_PARSER_ERROR(nvds_parse_3d_sink(sink, argv[1], "sink"));
    }
    }
/*------------------------------------------解析nvssink配置文件选项----------------------------------------*/     
  }
  if (PERF_MODE) {
      if(prop.integrated) {
          g_object_set (G_OBJECT (streammux), "nvbuf-memory-type", 4, NULL);
      } else {
          g_object_set (G_OBJECT (streammux), "nvbuf-memory-type", 2, NULL);
      }
  }
/*---------------------------------------------------parse the config file-------------------------------- */


/*---------------------------------------------------build the bus for pipeline-------------------------------- */
  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));

  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);
/*---------------------------------------------------build the bus for pipeline-------------------------------- */

/*---------------------------------------------------Add and link the elements in the pipeline -------------------------------- */
  /* we add all elements into the pipeline */
  // gst_bin_add_many (GST_BIN (pipeline), queue1, pgie, queue2,tiler,
  //     queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL);

  // /* we link the elements together */
  // if (!gst_element_link_many (streammux, queue1, pgie, queue2,tiler,
  //       queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL)) {
  //   g_printerr ("Elements could not be linked. Exiting.\n");
  //   return -1;
  // }
   /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline), queue1, streamdemux, tee_b0,tee_b1, queue_b0, queue_b1, queue_b2, queue_b3, streammux_b0, streammux_b1, pgie_b0, pgie_b1, metamuxer, queue2, tiler,
      queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL);

  /* we link the elements together */
  if (!gst_element_link (streammux, queue1))
  {
    g_printerr ("Elements streammux to queue1 could not be linked. Exiting.\n");
    return -1;
  }
  if (!gst_element_link (queue1, streamdemux))
  {
    g_printerr ("Elements queue1 to streamdemux could not be linked. Exiting.\n");
    return -1;
  }
  /*----------------------------link streamdemux to tee and queue in different branch----------------------------------------------*/
  // branch0
  link_element_to_demux_src_pad(streamdemux, tee_b0, 0);
  link_element_to_tee_src_pad (tee_b0, queue_b0);
  link_element_to_tee_src_pad (tee_b0, queue_b1);

  //branch1
  link_element_to_demux_src_pad(streamdemux, tee_b1, 1);
  link_element_to_tee_src_pad (tee_b1, queue_b2);
  link_element_to_tee_src_pad (tee_b1, queue_b3);

  /*----------------------------link streamdemux to tee and queue in different branch----------------------------------------------*/

  /*----------------------------link  different queue to one streammux in different branch----------------------------------------------*/
  // branch0
  link_element_to_streammux_sink_pad (streammux_b0, queue_b0, 0);
  link_element_to_streammux_sink_pad (streammux_b0, queue_b2, 1);
  //branch1
  link_element_to_streammux_sink_pad (streammux_b1, queue_b1, 0);
  link_element_to_streammux_sink_pad (streammux_b1, queue_b3, 1);
  /*----------------------------link  different queue to one streammux in different branch----------------------------------------------*/

  /*----------------------------link streammux to pgie in different branch----------------------------------------------*/
  if (!gst_element_link (streammux_b0, pgie_b0))
  {
    g_printerr ("Elements streammux_b0 to pgie_b0 could not be linked. Exiting.\n");
    return -1;
  }
  if (!gst_element_link (streammux_b1, pgie_b1))
  {
    g_printerr ("Elements streammux_b0 to pgie_b0 could not be linked. Exiting.\n");
    return -1;
  }
  /*----------------------------link streammux to pgie in different branch----------------------------------------------*/

  /*----------------------------link pgie to metamuxer in different branch----------------------------------------------*/
  link_element_to_metamux_sink_pad (metamuxer, pgie_b0, 0);
  link_element_to_metamux_sink_pad (metamuxer, pgie_b1, 1);
  if (!gst_element_link (metamuxer, queue2))
  {
    g_printerr ("Elements metamuxer to queue2 could not be linked. Exiting.\n");
    return -1;
  }

  /*----------------------------link pgie to metamuxer in different branch----------------------------------------------*/

  /*----------------------------link the rest element----------------------------------------------*/
  /* we link the elements together */
  if (!gst_element_link_many (queue2, tiler, queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  /*----------------------------link the rest element----------------------------------------------*/

/*---------------------------------------------------Add and link the elements in the pipeline -------------------------------- */

  /* Lets add probe to get informed of the meta data generated, we add probe to
   the sink pad of the osd element, since by that time, the buffer would have
   had got all the metadata. */
   /*让我们添加 probe来了解生成的元数据，我们将probe添加到osd元素的sink pad，因为到那时，缓冲区已经获得了所有元数据*/
  nvvidconv_sink_pad  = gst_element_get_static_pad (tiler, "src");
  if (!nvvidconv_sink_pad )   
    g_print ("Unable to get src pad\n");
  else
    gst_pad_add_probe (nvvidconv_sink_pad , GST_PAD_PROBE_TYPE_BUFFER,
        ronghe_tiler_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref (nvvidconv_sink_pad );

/*---------------------------------------------------Set the pipeline to "playing" state -------------------------------- */

  /* Set the pipeline to "playing" state */
  if (yaml_config) {
    g_print ("Using file: %s\n", argv[1]);
  }
  else {
    g_print ("Now playing:");
    for (i = 0; i < num_sources; i++) {
      g_print (" %s,", argv[i + 1]);
    }
    g_print ("\n");
  }
  //生成pipeline 拓扑图
  GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "dstest3-pipeline");
  // GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "dstest3-pipeline");
  //启动pipeline
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);
/*---------------------------------------------------Set the pipeline to "playing" state -------------------------------- */
/*---------------------------------------------------the pipeline to the end-------------------------------- */

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
/*---------------------------------------------------the pipeline to the end-------------------------------- */

  return 0;
}
