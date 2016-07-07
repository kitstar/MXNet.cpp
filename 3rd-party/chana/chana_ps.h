#pragma once

#include <functional>
#include <vector>
#include <cinttypes>
#include <string>
#include <list>

#define ALLREDUCE_IN_PLACE ((void *)-1)

#define ALLREDUCE_OP_SUM 0
#define ALLREDUCE_OP_UDR 1
#define ALLREDUCE_OP_NUM 2

#define ALLREDUCE_TYPE_INT32	0
#define ALLREDUCE_TYPE_UINT32	1
#define ALLREDUCE_TYPE_INT64	2
#define ALLREDUCE_TYPE_UINT64	3
#define ALLREDUCE_TYPE_FLOAT	4
#define ALLREDUCE_TYPE_DOUBLE	5
#define ALLREDUCE_TYPE_NUM		6

#define PS_ROLE_ALL             (-1)
#define PS_ROLE_COORDINATOR     (0)

#define PS_CMD_WAIT             (-1)
#define PS_CMD_WAIT_ACK         (-2)

# if defined(EXPORTDLL)
# if defined(_WIN32)
# define ExportDll __declspec(dllexport)
# else
# define ExportDll __attribute__((visibility("default")))
# endif
# else
# if defined(_WIN32)
# define ExportDll __declspec(dllimport)
# else
# define ExportDll
# endif
# endif

typedef void(*reducer_t)(void *state, void *opnd, size_t state_size);
typedef void(*broadcast_udf_t)(void *bcastbuf, size_t size, void *args);

typedef void(*ps_push_callback_t)(size_t count, uint64_t *keys, void **vals, size_t *val_sizes, void *args);
typedef void(*val_deallocator_t)(void *args);

class ExportDll ChaNaPSBase
{
public:
    ChaNaPSBase() { }

	// user defined interface
	//static ChaNaPSBase* Create();	

    virtual void control(int cmd_id, void *data, const size_t len) { }
    
    virtual void server_process_pull(
		uint64_t *keys,
		size_t key_count,
		void **vals,
		size_t *val_sizes,
		val_deallocator_t *dealloc,
		void **args,
		bool *fixed_val_size
		) = 0;

	virtual void server_process_push(size_t key_count, uint64_t *keys, void **vals, size_t *valsizes) = 0;
	virtual void worker_apply_pull(void *args) = 0;
};

typedef ChaNaPSBase*(*ps_create_function_t)(void *args);

ExportDll std::list<std::string> chana_config_get_string_value_list(const char* section, const char* key, char splitter, const char* dsptr);


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */ 

ExportDll void CreateSyncEngine(const char *machine_list_filename, const int iothread_per_machine_count, bool use_rdma);
ExportDll int GetMyRank();
ExportDll int GetMachineCount();
ExportDll size_t GetPSThreadNumPerMachine();
ExportDll void BarrierEnter();
ExportDll void ChaNa_AllReduce(const void *sendbuf, void *recvbuf, size_t elemcount, size_t elemsize, int elemtype, int op);
ExportDll void RegisterUserDefinedReducer(reducer_t reducer, int elemtype);

ExportDll void AsyncBroadCast(void *bcastbuf, size_t size);

// must perform barrier between this function and the following AsyncBroadCast()
ExportDll void RegisterAsyncBroadCastHandler(broadcast_udf_t bcast_cb, void *args);


/////////////////////// Parameter Server Interfaces ////////////////////////////////////////

ExportDll void CreateParameterServer(
	std::string machine_list_file,
	const int ps_per_machine_count,
	bool use_rdma,
	ps_create_function_t create_function,
	void *args
	);

ExportDll void CreateParameterServerWithPort(
	std::string machine_list_file,
	const int ps_per_machine_count,
	const int port,
	bool use_rdma,
	ps_create_function_t create_function,
	void *args
	);

ExportDll ChaNaPSBase * ChaNaPSGetInstance(int inst_id);
ExportDll void ChaNaPSWait();
ExportDll uint64_t ChaNaPSControl(int roles, void *data, size_t len, val_deallocator_t cb, void *args);
ExportDll uint64_t ChaNaPSPull(size_t count, uint64_t *keys, void **vals, size_t *val_sizes, void *args);
ExportDll uint64_t ChaNaPSPush(size_t count, uint64_t *keys, void **vals, size_t *val_sizes, ps_push_callback_t cb, void *args);

// init
ExportDll bool chana_initialize(int argc, const char *argv[]);
ExportDll bool chana_is_initialized(void);

// config
ExportDll const char* chana_config_get_value_string(const char* section, const char* key, const char* default_value, const char* dsptr);
ExportDll void chana_config_set(const char* section, const char* key, const char* value, const char* dsptr);
ExportDll bool chana_config_get_value_list(const char* section, const char* key, char splitter, const char* dsptr);
ExportDll bool chana_config_get_value_bool(const char* section, const char* key, bool default_value, const char* dsptr);
ExportDll uint64_t chana_config_get_value_uint64(const char* section, const char* key, uint64_t default_value, const char* dsptr);
ExportDll double chana_config_get_value_double(const char* section, const char* key, double default_value, const char* dsptr);
ExportDll int chana_config_get_all_keys(const char* section, const char** buffers, int buffer_count);
ExportDll bool chana_config_has_section(const char* section);
ExportDll bool chana_config_has_key(const char* section, const char* key);
ExportDll void chana_config_dump(const char* file);

#ifdef __cplusplus
}
#endif /* __cplusplus */