export NVDS_TEST3_PERF_MODE=0
sudo make
GST_DEBUG_DUMP_DOT_DIR=./dot/ ./deepstream-parallel-app dstest3_config.yml
cd ./dot
dot -Tpng dstest3-pipeline.dot > dstest3-pipeline.png