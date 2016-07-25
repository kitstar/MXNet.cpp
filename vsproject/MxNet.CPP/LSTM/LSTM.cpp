# include "lstm.h"

using namespace std;
using namespace mxnet::cpp;


LSTM::LSTM()
{
    learning_rate_ = chana_config_get_value_double(mxnet_section.c_str(), "learning_rate", 1e-4, "");
    weight_decay_ = chana_config_get_value_double(mxnet_section.c_str(), "weight_decay", 1e-4, "");    
    epoch_count_ = chana_config_get_value_uint64(mxnet_section.c_str(), "epoch_count", 25, "");
    batch_size_ = chana_config_get_value_uint64(mxnet_section.c_str(), "batch_size", 32, "");
    
    embedding_size_ = chana_config_get_value_uint64(mxnet_section.c_str(), "embedding_size", 200, "integer");
    num_lstm_layer_ = chana_config_get_value_uint64(mxnet_section.c_str(), "lstm_layer", 2, "integer");
    num_hidden_ = chana_config_get_value_uint64(mxnet_section.c_str(), "num_hidden", 200, "integer");
}


Symbol LSTM::BuildNetwork()
{
    return Symbol::Variable("A");
}


/* virtual */ void LSTM::Train(mxnet::cpp::KVStore *kv_store)
{
    string input_data = chana_config_get_value_string(mxnet_section.c_str(), "input_data", "", "");
    CHECK(!input_data.empty());

    string vocab_data = chana_config_get_value_string(mxnet_section.c_str(), "vocab_data", "", "");
    BuildVocabulary(vocab_data, input_data, vocab_map);
    
    /*

    unique_ptr<Optimizer> opt(new Optimizer("ccsgd", learning_rate_, weight_decay_));
    opt->SetParam("momentum", 0.9).SetParam("rescale_grad", 1.0 / (kv_store->GetNumWorkers() * batch_size_)).SetParam("clip_gradient", 10);

    if (running_mode_ == sync_mode_t::Async)
    {
        kv_store->SetOptimizer(std::move(opt));
        kv_store->Barrier();
    }

    auto lenet = BuildNetwork();

    // Simple Bind will initialize the parameter with SampleGaussian(0, 1)
    std::map<string, NDArray> grad_map;
    auto exe = lenet.SimpleBind(ctx_dev, args_map, grad_map, grad_type_map);

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
        auto dataReader = GetFileReader(input_image_data, batch_size_, 28 * 28, kv_store->GetRank(), kv_store->GetNumWorkers(), sizeof(uint32_t) * 4);
        auto labelReader = GetFileReader(input_label_data, batch_size_, 1, kv_store->GetRank(), kv_store->GetNumWorkers(), sizeof(uint32_t) * 2);

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
    */
}
