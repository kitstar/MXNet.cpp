# include "lstm.h"

using namespace std;
using namespace mxnet::cpp;


LSTM::LSTM()
{
    learning_rate_ = chana_config_get_value_double(mxnet_section.c_str(), "learning_rate", 1e-2, "");
    weight_decay_ = chana_config_get_value_double(mxnet_section.c_str(), "weight_decay", 1e-4, "");    
    epoch_count_ = chana_config_get_value_uint64(mxnet_section.c_str(), "epoch_count", 25, "");
    batch_size_ = chana_config_get_value_uint64(mxnet_section.c_str(), "batch_size", 32, "");
    momentum_ = chana_config_get_value_double(mxnet_section.c_str(), "momentum", 0, "double");
        
    num_lstm_layer_ = chana_config_get_value_uint64(mxnet_section.c_str(), "lstm_layer", 2, "integer");
    hidden_size_ = chana_config_get_value_uint64(mxnet_section.c_str(), "num_hidden", 200, "integer");
    num_steps_ = chana_config_get_value_uint64(mxnet_section.c_str(), "num_step", 0, "integer");
    embedding_size_ = chana_config_get_value_uint64(mxnet_section.c_str(), "embedding_size", 200, "integer");

    unknown_token_ = chana_config_get_value_string(mxnet_section.c_str(), "unknown_token", "<unk>", "string");
    drop_out_ = chana_config_get_value_double(mxnet_section.c_str(), "dropout", 0, "real");
}


LSTM::LstmState LSTM::Cell(
    int num_hidden, const Symbol &indata, const LstmState &prev_state, 
    const LstmParam &param, int seq_idx, int layer_idx, double dropout)
{
    string name = " (l" + to_string(layer_idx) + " t" + to_string(seq_idx) + ")";

    auto pindata = indata;

    if (dropout > DBL_EPSILON)
    {
        pindata = Dropout("DropOut Input" + name, pindata, dropout);
    }
    
    // for cell    
    auto i2h = FullyConnected("i2h" + name, pindata, param.i2h_weight, param.i2h_bias, num_hidden * 4);      
    auto h2h = FullyConnected("h2h" + name, prev_state.h, param.h2h_weight, param.h2h_bias, num_hidden * 4);

    // input gate, input transform, forget gate, output gate
    auto gates = i2h + h2h;
    auto slice_gates = SliceChannel("Slice Gates" + name, gates, 4);
    auto in_gate = Activation("Input Gate" + name, slice_gates[0], ActivationActType::sigmoid);
    auto in_transform = Activation("Input Transform" + name, slice_gates[1], ActivationActType::tanh);
    auto forget_gate = Activation("Forget Gate" + name, slice_gates[2], ActivationActType::sigmoid);
    auto output_gate = Activation("Output Gate" + name, slice_gates[3], ActivationActType::sigmoid);
    LstmState next_state("h" + name, "c" + name);
    next_state.c = (forget_gate * prev_state.c) + (in_gate * in_transform);
    next_state.h = output_gate * Activation("Transform H" + name, next_state.c, ActivationActType::tanh);

    return next_state;
}


Symbol LSTM::BuildNetwork()
{
    // Input Words (batch_size * num_steps)
    auto input_data = Symbol::Variable("Data");
    args_map["Data"] = NDArray(Shape(batch_size_, num_steps_), ctx_cpu, false);
    grad_type_map["Data"] = kNullOp;

    // Embedding Layer    
    auto embed_weight = Symbol::Variable("embed_weight");
    //args_map["embed_weight"] = NDArray(Shape(vocab_size_, embedding_size_), ctx_dev, false);
    auto embed = Embedding("Embedding", input_data, embed_weight, vocab_size_, embedding_size_);
    auto wordvec = SliceChannel("WordVec", embed, num_steps_);

    // Init hidden state
    vector<LstmState> current_state;
    vector<LstmParam> cell_param;
    for (int layer = 0; layer < num_lstm_layer_; ++layer)
    {        
        current_state.push_back(LstmState("init h (l" + to_string(layer) + ")", "init c (l" + to_string(layer) + ")"));
        // args_map["init h(l" + to_string(layer) + ")"] = NDArray(Shape(batch_size_, num_steps_), ctx_cpu, false);
        cell_param.push_back(LstmParam(" (l" + to_string(layer) + ")"));
    }

    // Step LSTM
    vector<Symbol> hidden_all;
    for (int seqidx = 0; seqidx < num_steps_; ++seqidx)
    {
        double dp_ratio = 0;
        auto hidden = wordvec[seqidx];
        // Stack LSTM
        for (int layer = 0; layer < num_lstm_layer_; ++layer)
        {
            if (layer == 0) dp_ratio = 0;
            else dp_ratio = drop_out_;
           
            auto next_state = Cell(hidden_size_, hidden, current_state[layer], cell_param[layer], seqidx, layer, dp_ratio);
            hidden = next_state.h;
            current_state[layer] = next_state;
        }
        if (drop_out_ > DBL_EPSILON)
        {
            hidden = Dropout("drop out hidden l" + to_string(seqidx), hidden, drop_out_);
        }
        hidden_all.push_back(hidden);
    }
    auto hidden_concat = Concat("hidden concat", hidden_all, num_steps_);
    
    auto cls_weight = Symbol::Variable("cls weight");
    auto cls_bias = Symbol::Variable("cls bias");
    auto pred = FullyConnected("predict", hidden_concat, cls_weight, cls_bias, vocab_size_);
    
    auto label = Symbol::Variable("Label");
    // transpose("label trans", label, label);
    // label = Reshape("label reshape", label, Shape(1));

    auto softmax = SoftmaxOutput("softmax", pred, label);

    auto rnn = current_state[0].c;

    rnn.InferArgsMap(ctx_dev, &args_map, args_map);
    for (auto &s : rnn.ListArguments())
    {
        auto &e = args_map[s];
        cout << s << " : ";
        cout << e.GetShape().at(0);
        for (size_t it = 1; it < e.GetShape().size(); ++it)
            cout << " * " << e.GetShape().at(it);
        cout << endl;
    }

    return rnn;
}


/* virtual */ void LSTM::Train(mxnet::cpp::KVStore *kv_store)
{
    CHECK(!unknown_token_.empty());
    CHECK_GT(num_steps_, 0);
    
    string input_data = chana_config_get_value_string(mxnet_section.c_str(), "input_data", "", "");
    CHECK(!input_data.empty());

    string vocab_data = chana_config_get_value_string(mxnet_section.c_str(), "vocab_data", "", "");
    CHECK(!vocab_data.empty());
    vocab_size_ = BuildVocabulary(vocab_data, input_data, vocab_map, unknown_token_);
    CHECK_NE(vocab_size_, 0);
        
    unique_ptr<Optimizer> opt(new Optimizer("ccsgd", learning_rate_, weight_decay_));
    opt->SetParam("momentum", momentum_).SetParam("rescale_grad", 1.0 / (kv_store->GetNumWorkers() * batch_size_)).SetParam("clip_gradient", 10);

    if (running_mode_ == sync_mode_t::Async)
    {
        kv_store->SetOptimizer(std::move(opt));
        kv_store->Barrier();
    }

    auto rnn = BuildNetwork();

    // Simple Bind will initialize the parameter with SampleGaussian(0, 1)
    std::map<string, NDArray> grad_map;
    auto exe = rnn.SimpleBind(ctx_dev, args_map, grad_map, grad_type_map);

    auto start_time = std::chrono::system_clock::now();

    vector<NDArray> parameters;
    vector<NDArray> gradients;

    {
        auto arguments = rnn.ListArguments();
        for (size_t idx = 0; idx < arguments.size(); ++idx)
        {
            auto &name = arguments[idx];
            if (name.substr(0, 4) != "Data")
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
                for (size_t idx = 0; idx < parameters.size(); ++idx)
                    NDArray::SampleGaussian(0, 0.01, &parameters[idx]);
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
    /*
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
