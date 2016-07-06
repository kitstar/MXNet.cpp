# include "mnist.h"
# include "kvstore.h"

using namespace std;
using namespace mxnet::cpp;


Mnist::Mnist()
{
    learning_rate_ = chana_config_get_value_double(mxnet_section.c_str(), "learning_rate", 1e-4, "");
    weight_decay_ = chana_config_get_value_double(mxnet_section.c_str(), "weight_decay", 1e-4, "");
    image_size_ = chana_config_get_value_uint64(mxnet_section.c_str(), "image_size", 28 * 28, "");    
    epoch_count_ = chana_config_get_value_uint64(mxnet_section.c_str(), "epoch_count", 1, "");
    batch_size_ = chana_config_get_value_uint64(mxnet_section.c_str(), "batch_size", 100, "");
}

/* virtual */ Symbol Mnist::BuildNetwork()
{
    auto data = Symbol::Variable("Data");
    args_map["Data"] = NDArray(Shape(batch_size_, 1, 28, 28), ctx_cpu, false);    
    
    // first conv    
    auto conv1_w = Symbol::Variable("conv1 weight");    
    auto conv1_b = Symbol::Variable("conv1 bias");    
    auto conv1 = Convolution("Conv1", data, conv1_w, conv1_b, Shape(5, 5), 20);
    auto tanh1 = Activation("Convolution1 activation", conv1, ActivationActType::tanh);
    auto pool1 = Pooling("Convolution1 pooling", tanh1, Shape(2, 2), PoolingPoolType::max, Shape(2, 2));
    
    // second conv    
    auto conv2_w = Symbol::Variable("conv2 weight");    
    auto conv2_b = Symbol::Variable("conv2 bias");    
    auto conv2 = Convolution("Conv2", pool1, conv2_w, conv2_b, Shape(5, 5), 50);
    auto tanh2 = Activation("Convolution2 activation", conv2, ActivationActType::tanh);
    auto pool2 = Pooling("Convolution2 pooling", tanh2, Shape(2, 2), PoolingPoolType::max, Shape(2, 2));    
        
    // first fullc    
    auto flatten = Flatten("Flatten", pool2);
    auto fc1_w = Symbol::Variable("FC1 weight");
    auto fc1_b = Symbol::Variable("FC1 bias");
    auto fc1 = FullyConnected("FC1", flatten, fc1_w, fc1_b, 500);
    auto tanh3 = Activation("FC1 activation", fc1, ActivationActType::tanh);
    
    // second fullc
    auto fc2_w = Symbol::Variable("FC2 weight");
    auto fc2_b = Symbol::Variable("FC2 bias");
    auto fc2 = FullyConnected("FC2", tanh3, fc2_w, fc2_b, 10);        
    
    // loss
    auto label = Symbol::Variable("Label");    
    auto lenet = SoftmaxOutput("Lenet output", fc2, label);
    lenet.InferArgsMap(ctx_dev, &args_map, args_map);

# if !defined(NDEBUG)
    for (auto &s : lenet.ListArguments())
    {
        auto &e = args_map[s];
        cout << s << " : " << e.GetShape().at(0);
        for (size_t it = 1; it < e.GetShape().size(); ++it)
            cout << " * " << e.GetShape().at(it);
        cout << endl;
    }
# endif
    return lenet;
}


/* virtual */ size_t Mnist::CorrectCount(const mxnet::cpp::NDArray &result, const mxnet::cpp::NDArray &labels, size_t sample_count)
{    
    // For Softmax output    
    result.WaitToRead();
    int cat_num = result.GetShape()[1];    
    size_t nCorrect = 0;
            
    const mx_float *pResult = result.GetData();
    const mx_float *pLabel = labels.GetData();
    for (int i = 0; i < sample_count; ++i)
    {
        auto prediction_label = GetSoftmaxResult(pResult + i * cat_num, cat_num);
        if (abs(pLabel[i] - prediction_label) < FLT_EPSILON) nCorrect++;
    }    
 
    return nCorrect;
}

/* virtual */ void Mnist::PrintResult(dmlc::Stream *stream, const mxnet::cpp::NDArray &result, size_t sample_count)
{
    // For Softmax    
    result.WaitToRead();
    int cat_num = result.GetShape()[1];    
    mx_float *temp_store = (mx_float *)malloc(sample_count * sizeof(mx_float));

    const mx_float *pResult = result.GetData();    
    for (int i = 0; i < sample_count; ++i)
        temp_store[i] = GetSoftmaxResult(pResult + i * cat_num, cat_num);

    stream->Write(temp_store, sample_count * sizeof(mx_float));
    free(temp_store);
}


/* virtual */ void Mnist::Train(KVStore *kv_store)
{
    string input_image_data = chana_config_get_value_string(mxnet_section.c_str(), "image_data", "", "");
    CHECK(!input_image_data.empty());

    string input_label_data = chana_config_get_value_string(mxnet_section.c_str(), "label_data", "", "");
    CHECK(!input_label_data.empty());

    unique_ptr<Optimizer> opt(new Optimizer("ccsgd", learning_rate_, weight_decay_));
    opt->SetParam("momentum", 0.9).SetParam("rescale_grad", 1.0 / (kv_store->GetNumWorkers() * batch_size_)).SetParam("clip_gradient", 10);

    if (running_mode_ == sync_mode_t::Async)
    {
        kv_store->SetOptimizer(std::move(opt));
        kv_store->Barrier();
    }

    auto lenet = BuildNetwork();

    // Simple Bind will initialize the parameter with SampleGaussian(0, 1)
    auto exe = lenet.SimpleBind(ctx_dev, args_map);

    auto start_time = std::chrono::system_clock::now();
    
    vector<NDArray> parameters;
    vector<NDArray> gradients;

    {
        auto arguments = lenet.ListArguments();
        for (size_t idx = 0; idx < arguments.size(); ++idx)
        {
            auto &name = arguments[idx];
            if (name != "Data" && name != "Label")
            {
                parameters.push_back(exe->arg_arrays[idx]);
                gradients.push_back(exe->grad_arrays[idx]);
            }
            else exe->
        }
    }

    string input_model_file = chana_config_get_value_string(mxnet_section.c_str(), "input_model", "", "");
    if (input_model_file.empty())
    {
        // Initialize Parameters
        if (running_mode_ != sync_mode_t::Sync)
        {
            for (size_t idx = 0; idx < parameters.size(); ++idx)
            {
                parameters[idx].WaitToRead();
                kv_store->Init(idx, parameters[idx]);
                kv_store->Pull(idx, &parameters[idx]);
            }
        }
        else
        {
            if (kv_store->GetRank() == 0)
            {
                // For Master, initialize the weights and bias if you need.
            }
            else
            {
                // For Slaves
                for (size_t idx = 0; idx < parameters.size(); ++idx)
                    parameters[idx] = 0;
            }

            NDArray::WaitAll();
            kv_store->AllReduce(&parameters);
        }
    }
    else
    {
        LoadModel(input_model_file, parameters);
    }
        
    // Start Training
    size_t processed_sample_count = 0;
    for (int current_epoch = 0; current_epoch < epoch_count_; ++current_epoch)
    {
        // Skip Header Information for mnist dataset.
        auto dataReader = get_file_reader(input_image_data, batch_size_, 28 * 28, kv_store->GetRank(), kv_store->GetNumWorkers(), sizeof(uint32_t) * 4);
        auto labelReader = get_file_reader(input_label_data, batch_size_, 1, kv_store->GetRank(), kv_store->GetNumWorkers(), sizeof(uint32_t) * 2);
        
        while (!dataReader->Eof())
        {
            auto data = dataReader->ReadBatch();
            size_t readed_sample_count = data.size() / image_size_;            
            auto label = labelReader->ReadBatch();
            CHECK_EQ(readed_sample_count, label.size());

            args_map["Data"].SyncCopyFromCPU(data.data(), batch_size_ * image_size_);
            args_map["Label"].SyncCopyFromCPU(label.data(), batch_size_);
            NDArray::WaitAll();
            
            exe->Forward(true);
            exe->Backward();           
            
            if (running_mode_ != sync_mode_t::Sync)
            {
                // Parameter Server Method
                for (size_t idx = 0; idx < parameters.size(); ++idx)
                {
                    kv_store->Push(idx, gradients[idx]);
                    kv_store->Pull(idx, &parameters[idx]);
                }
            }
            else
            {
                // Allreduce Method
                kv_store->AllReduce(&gradients);
                for (size_t idx = 0; idx < parameters.size(); ++idx)
                {
                    gradients[idx].WaitToRead();
                    parameters[idx].WaitToWrite();
                    opt->Update(idx, parameters[idx], gradients[idx]);
                }
            }
            
            processed_sample_count += readed_sample_count;
            auto cur_time = std::chrono::system_clock::now();
            auto elapse_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - start_time);
                        
            auto count = CorrectCount(exe->outputs[0], args_map["Label"], readed_sample_count);
            LG << "Iter " << current_epoch
                << ", accuracy: " << count * 100.0 / readed_sample_count << "%"
                << "\tsample/s: " << processed_sample_count * 1000.0 / elapse_in_ms.count()
                << "\tProgress: [" << processed_sample_count * 100.0 / epoch_count_ / dataReader->recordCount() << "%]";            
        }

        delete dataReader;
        delete labelReader;
    }

    kv_store->Barrier();

    auto cur_time = std::chrono::system_clock::now();
    auto elapse_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - start_time);
    LG << "Total Sample: [" << processed_sample_count << "]"
        << "\tSpeed: [" << processed_sample_count * 1000.0 / elapse_in_ms.count() << "/s]"
        << "\tTime: [" << elapse_in_ms.count() / 1000.0 << "s]";

    if (kv_store->GetRank() == 0)
    {
        if (running_mode_ == sync_mode_t::Async)
            for (size_t idx = 0; idx < parameters.size(); ++idx)            
                kv_store->Pull(idx, &parameters[idx]);
        
        string output_model_name = chana_config_get_value_string(mxnet_section.c_str(), "output_model", "", "");
        if (!output_model_name.empty())
        {
            SaveModel(output_model_name, parameters);
        }
    }

    kv_store->Barrier();

    delete exe;
}


/* virtual */ void Mnist::Predict(KVStore *kv_store, bool is_validation)
{
    string input_image_data = chana_config_get_value_string(mxnet_section.c_str(), "image_data", "", "");
    CHECK(!input_image_data.empty());
    
    string input_label_data = chana_config_get_value_string(mxnet_section.c_str(), "label_data", "", "");
    if (is_validation) CHECK(!input_label_data.empty());

    // You can choice a suitable batch size for prediction.
    batch_size_ = 8 * 1024 * 1024 / image_size_;

    auto lenet = BuildNetwork();    
    auto exe = lenet.SimpleBind(ctx_dev, args_map);

    auto start_time = std::chrono::system_clock::now();

    vector<NDArray> parameters;
    vector<NDArray> gradients;

    {
        auto arguments = lenet.ListArguments();
        for (size_t idx = 0; idx < arguments.size(); ++idx)
        {
            auto &name = arguments[idx];
            if (name != "Data" && name != "Label")
            {
                parameters.push_back(exe->arg_arrays[idx]);
                gradients.push_back(exe->grad_arrays[idx]);
            }
        }
    }

    string input_model_file = chana_config_get_value_string(mxnet_section.c_str(), "input_model", "", "");
    CHECK(!input_model_file.empty());    
    LoadModel(input_model_file, parameters);    

    string output_label_file = chana_config_get_value_string(mxnet_section.c_str(), "output_label", "", "");
    dmlc::Stream *output_stream = nullptr;
    if (!output_label_file.empty())
    {
        LG << "Save predicting result at [" << output_label_file << "]";
        output_stream = dmlc::Stream::Create(output_label_file.c_str(), "w", true);
    }

    // Start Prediction    
    auto data_reader = get_file_reader(input_image_data, batch_size_, 28 * 28, kv_store->GetRank(), kv_store->GetNumWorkers(), sizeof(uint32_t) * 4);
    DataReader *label_reader = nullptr;
    if (is_validation)
    {
        label_reader = get_file_reader(input_label_data, batch_size_, 1, kv_store->GetRank(), kv_store->GetNumWorkers(), sizeof(uint32_t) * 2);
    }

    size_t processed_sample_count = 0;
    size_t correct_count = 0;
    while (!data_reader->Eof())
    {
        auto data = data_reader->ReadBatch();
        size_t readed_sample_count = data.size() / image_size_;
        args_map["Data"].SyncCopyFromCPU(data.data(), batch_size_ * image_size_);
            
        std::vector<float> label;
        if (is_validation)
        {
            label = label_reader->ReadBatch();
            CHECK_EQ(readed_sample_count, label.size());
            args_map["Label"].SyncCopyFromCPU(label.data(), batch_size_);
        }
        NDArray::WaitAll();

        exe->Forward(false);

        if (nullptr != output_stream) PrintResult(output_stream, exe->outputs[0], readed_sample_count);
                    
        processed_sample_count += readed_sample_count;
        auto cur_time = std::chrono::system_clock::now();
        auto elapse_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - start_time);

        if (is_validation)
        {
            auto count = CorrectCount(exe->outputs[0], args_map["Label"], readed_sample_count);
            correct_count += count;
            LG << "Accuracy: " << count * 100.0 / readed_sample_count << "%";
        }
        LG << "Sample/s: [" << processed_sample_count * 1000.0 / elapse_in_ms.count() << "]"
            << "\tProgress: [" << processed_sample_count * 100.0 / data_reader->recordCount() << "%]";
    }

    delete data_reader;
    if (label_reader != nullptr) delete label_reader;
    if (nullptr != output_stream)
    {
        delete output_stream;
    }
        
    kv_store->Barrier();

    auto cur_time = std::chrono::system_clock::now();
    auto elapse_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - start_time);
    LG << "Total Sample: [" << processed_sample_count << "]"
        << "\tSpeed: [" << processed_sample_count * 1000.0 / elapse_in_ms.count() << "/s]"
        << "\tTime: [" << elapse_in_ms.count() / 1000.0 << "s]";
    if (is_validation)
    {
        LG << "Final Accuracy : [" << correct_count * 100.0 / processed_sample_count << "%]";
    }

    delete exe;
}


int main(int argc, const char *argv[])
{
    LG << "Usage:";
    LG << "\tFor distributed training: " << argv[0] << " machine_list  server_count_per_machine" << endl;    

    auto chana_config = getenv("CHANA_CONFIG_FILE");
    if (chana_config != nullptr)
    {
        std::string config(chana_config);
        config = "config=" + config;
        const char *temp_argv[] = { argv[0], config.c_str() };
        chana_initialize(2, temp_argv);
    }
    else
    {
        std::string config("config=config.ini");
        const char *temp_argv[] = { argv[0], config.c_str() };
        chana_initialize(2, temp_argv);        
    }
        
    Mnist mnist;
    KVStore *kv_store = nullptr;
    if (argc > 2) kv_store = Mlp::InitializeKvstore(mnist.running_mode_, argv[1], argv[2]);
    else kv_store = Mlp::InitializeKvstore(mnist.running_mode_, "", "");

    mnist.Run(kv_store);
   
    return 0;
}