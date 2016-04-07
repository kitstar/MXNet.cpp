# pragma once

# include <mutex>
# include <string>
# include "util.h"
# include "data.h"
# include "io/filesys.h"
# include "MxNetCpp.h"


void init_env()
{
    std::string entry = "CLASSPATH=";

    // Init classpath
    char buf[129];
    FILE* output = _popen("hadoop classpath --glob", "r");
    while (true)
    {
        size_t len = fread(buf, sizeof(char), sizeof(buf) - 1, output);
        if (len == 0)
            break;
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


class Mlp 
{
public:
    Mlp(bool _is_local) : ctx_cpu(mxnet::cpp::Context(mxnet::cpp::DeviceType::kCPU, 0)),
        ctx_dev(mxnet::cpp::Context(mxnet::cpp::DeviceType::kCPU, 0)),
        is_local_data(_is_local)
    { }
    
    void Run(mxnet::cpp::KVStore *kv, std::unique_ptr<dmlc::SeekStream> stream, size_t streamSize, bool sync);	

protected:
    void output_model();

    float ValAccuracy(mxnet::cpp::Symbol mlp,
        const mxnet::cpp::NDArray& samples,
        const mxnet::cpp::NDArray& labels);

    float Auc(const mxnet::cpp::NDArray& result, const mxnet::cpp::NDArray& labels);


private:
    const static int batchSize = 300;
    const static int sampleSize = 601;
    const static int maxEpoch = 5;
    
    mxnet::cpp::Context ctx_cpu;
    mxnet::cpp::Context ctx_dev;
    std::map<std::string, mxnet::cpp::NDArray> args_map;
	std::map<std::string, mxnet::cpp::NDArray> grads_map;
	std::map<std::string, mxnet::cpp::OpReqType> reqtype_map;

	std::vector<mxnet::cpp::NDArray> in_args;
	std::vector<mxnet::cpp::NDArray> arg_grad_store;
	std::vector<mxnet::cpp::OpReqType> grad_req_type;

    float learning_rate = 0.01;
    float weight_decay = 1e-5;
    bool is_local_data;
};