# include "util.h"

void GetFileStream(const std::string &file_name, dmlc::Stream *&stream, size_t *file_size, const char *op)
{
    if (strchr(op, 'r') != nullptr)
    {
        dmlc::io::URI file_uri(file_name.c_str());
        auto fs = dmlc::io::FileSystem::GetInstance(file_uri);
        *file_size = fs->GetPathInfo(file_uri).size;
        stream = fs->OpenForRead(file_uri, false);
    }
    else
    {
        file_size = 0;
        stream = dmlc::Stream::Create(file_name.c_str(), op, true);
    }
}


void MergeFiles(std::string final_name, int count)
{
    dmlc::Stream *final_stream = nullptr;
    size_t file_size;
    size_t readed_size = 0;
    const size_t buf_size = 64 * 1024 * 1024;
    char *buf = (char *)malloc(buf_size + 64);

    GetFileStream(final_name, final_stream, &file_size, "w");

    for (int i = 0; i < count; ++i)
    {
        std::string cur_file = final_name + ".part" + std::to_string(i);
        dmlc::Stream *cur_stream = nullptr;
        GetFileStream(cur_file, cur_stream, &file_size, "r");

        while ((readed_size = cur_stream->Read(buf, buf_size)) > 0)
        {
            final_stream->Write(buf, readed_size);
        }
        delete cur_stream;
    }

    free(buf);
    delete(final_stream);
}


DataReader * GetFileReader(const std::string &file_name, int buffer_sample_count, int sample_size, int my_rank, int total_rank, int skip_offset)
{
    if (file_name.substr(0, 7) != "hdfs://")
    {
        my_rank = 0;
        total_rank = 1;
    }

    dmlc::Stream *stream = nullptr;
    size_t file_size = 0;
    GetFileStream(file_name, stream, &file_size, "r");
    auto ret = new DataReader(dynamic_cast<dmlc::SeekStream *>(stream), file_size, sample_size, my_rank, total_rank, buffer_sample_count, skip_offset);
    return ret;
}


void init_env(std::string file_name)
{
    if (file_name.substr(0, 7) == "hdfs://")
    {
        std::string entry = "CLASSPATH=";

        // Init classpath
        char buf[129];
        FILE* output = _popen("hadoop classpath --glob", "r");
        while (true)
        {
            size_t len = fread(buf, sizeof(char), sizeof(buf) - 1, output);
            if (len == 0) break;
            buf[len] = 0;
            entry += buf;
        }
        fclose(output);
        entry.pop_back(); // Remove line ending
        _putenv(entry.c_str());

# if !defined(NDEBUG)
        LG << entry;
# endif
    }
}