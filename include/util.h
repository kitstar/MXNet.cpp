# pragma once

# include <chrono>
# include <string>
# include "../src/io/filesys.h"
# include "data.h"


inline double get_time()
{
  using namespace std::chrono;
  high_resolution_clock::duration tp = high_resolution_clock::now().time_since_epoch();
  return (double)tp.count() * high_resolution_clock::period::num / high_resolution_clock::period::den;
}

void GetFileStream(const std::string &file_name, dmlc::Stream *&stream, size_t *file_size, const char *op);

DataReader * GetFileReader(const std::string &file_name, int buffer_sample_count, int sample_size, int my_rank, int total_rank, int skip_offset = 0);

void MergeFiles(std::string final_name, int count);

void init_env(std::string file_name);