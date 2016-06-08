# pragma once

# include "chana_ps.h"
# include "MLP.h"


class Seq2seq : public Mlp
{
public:
    Seq2seq();

    virtual size_t train(std::string file_name, std::string kvstore_args) { return 0; }

    virtual size_t predict(std::string file_name, std::string model_name, std::string kvstore_args) { return 0; }

protected:
    virtual void build_network() { }

    virtual void output_model(std::string model_name) { }

    virtual void load_model(std::string model_name) { }    


private:
    std::vector<mxnet::cpp::NDArray> in_args;
    std::vector<mxnet::cpp::NDArray> arg_grad_store;
    std::vector<mxnet::cpp::OpReqType> grad_req_type;

    const int vocab_size = 40000;
    const int embedding_size = 200;
    const int encoder_hid_dim = 500;
    const int decoder_hid_dim = 1000;
    const double init_weight_scale = 0.01;

    std::string data_folder = "/media/xiaso/Data/skip-thoughts/";
    std::string path_to_vocab = data_folder + "\\blocks_query_encoder\\model\\vocab.pkl";
    std::string path_to_model = data_folder + "\\blocks_query_encoder\\model\\preserved3\\";
    std::string path_to_word2vec = data_folder + "\\glove\\vectors.txt.pkl";

};