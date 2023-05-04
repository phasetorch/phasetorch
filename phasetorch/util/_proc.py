import multiprocessing as mp
import os
import numpy as np

# VS: The routine launch_proc() is invoked by MultiProc.launch()
#     launch_proc() specifies the computation to be executed by each process.
#     The group or pool of processes is spawned by MultiProc.launch()

# VS: Below routine is executed by a single process.
#     estor represents phase-retrieval method such as NLPR or paganin.
#     init_args are parameters for estor such as  (device <CPU/GPU>, solver_kwargs <LBFGS optimizer options>, tran_func <transfer function handle such as NonLinTran or ConNonLinTran>, Nviews <usualy set to 1>, Npad_y, Npad_x <detector plane padding>,
#                                                   .... pix_wid, energy, prop_dists, mag_fact, reg_par)
#     run_args is a list of tuples (typically just 1 tuple) , where each tuple contains the data-arrays for processing (single-view radiograph, corresponding weights, corresponding output array)
def launch_proc(pid,device,estor,init_args,run_args):
    if device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = pid

    # VS: pretty neat that python allows the name of the class to be treated as a variable.
    # For eg: define a class named "MyClass" whose constructor __init__ takes 2 input arguments.
    # Then, the line "classname = MyClass", followed by the line "classname(arg1, agr2)" works !! it defines an object of type "MyClass"
    # VS: The use of * is to separate out of multiple elemnts in a tuple or list into separate elements.
    pr = estor(*init_args)
    rvals = []
    # VS: Typically run_args is a single tuple containing (radiograph for single view, associated weights, associated output array) ...
    #     But in more general case run_args can be a list of more than 1 tuple, meaning multiple views to be processed by a single device/process.
    for arg in run_args:
        rvals.append(pr.run(*arg))
    print('Finished job on process {}'.format(pid))
    return rvals

class MultiProc:
    # VS: The use of argument "args" in __init__() and queue() are very different.
    # In __init__() args referers to : (device <CPU/GPU>, solver_kwargs <LBFGS optimizer options>, tran_func <transfer function handle such as NonLinTran or ConNonLinTran>, Nviews <usualy set to 1>, Npad_y, Npad_x <detector plane padding>,
    #                                   .... pix_wid, energy, prop_dists, mag_fact, reg_par)
    # VS: For some reason "device" is passed twice, once in args and once outside it.
    # In queue(), args refers to the data that needs to be processed ...
    # (radiograph, data-weighting, and initial estimate of the transmission field that is to be reconstructed for a given view)
    def __init__(self,estor,args,processes,device):
        # Here estor refers to phase-retrieval method such as NLPR, Paganin, etc.
        self.estor = estor
        self.init = args
        self.processes = processes
        self.device = device
        self.run = []

    # VS: Typically for each process (more generally for each task where #tasks > #processes), the data to be utilized is inserted into the queue (implemented as a list here) one at a time.
    def queue(self,args):
        self.run.append(args)

    # VS: added this in case queue needs to be repeatedly cleared and updated/launched.
    def clear_queue(self):
        self.run = []

    def launch(self):
        # VS: Typically jobs_tot which represents number of "tasks" in the queue will be set to #processes, though in general any value is allowed.
        jobs_tot = len(self.run)
        # VS: Typically jobs_proc will be 1 in the above case.
        jobs_proc = jobs_tot/self.processes
        jobs_proc = int(np.ceil(jobs_proc))
        pool_args = []
        for pid in range(self.processes):
            vmin = pid*jobs_proc
            vmax = (pid+1)*jobs_proc
            vmax = jobs_tot if vmax>jobs_tot else vmax
            args = (str(pid),self.device,self.estor,self.init,self.run[vmin:vmax])
            # VS: Even in the case where jobs_tot > #processes, the list pool_args will be of length #processes.
            #     pool_args represents the data and arguments for each process.
            pool_args.append(args)

        # VS: Basic of multiprocessing in python. There are 3 important steps / commands:
        # 1) Optional first line (recommended): Use mp.set_start_method('spawn'). The other option is to use 'fork', but 'spawn' though slower is safer.
        # 2) p = mp.Pool(num_processes) will create a group of processes denoted by "p".
        # 3) Then, out_list = p.map(func_handle, arg_list) where arg_list is of size num_processes will launch parallel processes, each based on the same function specified by func_handle but on a different input argument.
        #    The output of each process is collectively returned as a list denoted by "out_list".

        # VS: Some modifications to the above basics of multiprocessing that are used here.
        # For step 1), Optional modification: You can write ctx = mp.get_context('spawn') and then use ctx as an object that has all properites of mp, such as ctx.Pool instead of mp.Pool and so on.
        # For step 3), required modification: Use starmap() instead of map() to assign arguments to each process.
        #              This allows us to use tuples in the list of arguments, i.e. starmap(func_handle, arglist = [tuple_1, tuple_2, ..., tuple_N ] ) where each tuple denotes the collection of inputs for a given process.

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=self.processes) as pool:
            pool_rvals = pool.starmap(launch_proc,pool_args)

        rvals = []
        # VS: list.extend(arg) will break arg into individual members and then append each member.
        # VS: Note in the below line that rvals.extend is used instead of rvals.append to work for most general case where each process handles a different number of tasks (i.e. jobs_tot is not a perfect multiple of #processes.)
        for rv in pool_rvals:
            rvals.extend(rv)
        return rvals

    def launch_onebyone(self):
        rvals = []
        seq_inst = len(self.run)/self.processes
        seq_inst = int(np.ceil(seq_inst))
        for inst in range(seq_inst):
            num_procs = 0
            pool_args = []
            for pid in range(self.processes):
                vmin = inst*self.processes+pid
                vmax = vmin+1
                if vmax > len(self.run):
                    break
                else:
                    num_procs = num_procs+1
                args = (str(pid), self.device, self.estor, self.init, self.run[vmin:vmax])
                pool_args.append(args)

            print('Running {} process instances from {}'.format(self.processes, inst*self.processes))
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=self.processes) as pool:
                pool_rvals = pool.starmap(launch_proc, pool_args)

            for rv in pool_rvals:
                rvals.extend(rv)
            # VS: Added a line temporarity to track progress
            print("Multi-processing: Progress = {} {}".format(round((100*(inst+1)/seq_inst), 2), '%'))

        assert vmax >= len(self.run)
        return rvals

    #VS: Added a simpler and very specific implementation for the case where #processes = length of queue (self.run).
    #    In this implementation, the queue self.run is a master-list where each member of the master-list is a list of individual tuples, ...
    #    where each tuple represents the arguments for a single execution of self.estor.run() where self.estor is a class.
    # VS: introduced the sub_launches variable to handle situations where multiprocessing.pool() is stressed or fails to pool when number of arguments is too many.
    def launch_onebyone_simple(self, sub_launches=1):
        assert(self.processes == len(self.run))
        rvals = []

        if(sub_launches==1):
            pool_args = []
            for pid in range(self.processes):
                args =  (str(pid), self.device, self.estor, self.init, self.run[pid])
                pool_args.append(args)

            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=self.processes) as pool:
                pool_rvals = pool.starmap(launch_proc, pool_args)
        else:
            pool_rvals = [ [] for  _ in range(self.processes)]
            ctx = mp.get_context('spawn')

            for idx_launch in range(sub_launches):
                pool_args = []
                for pid in range(self.processes):
                    start = (idx_launch*len(self.run[pid]))//sub_launches
                    stop  = ((idx_launch+1)*len(self.run[pid]))//sub_launches
                    args  =  (str(pid), self.device, self.estor, self.init, self.run[pid][start:stop])
                    pool_args.append(args)

                with ctx.Pool(processes=self.processes) as pool:
                    rvals_temp =  pool.starmap(launch_proc, pool_args)

                for pid in range(self.processes):
                    pool_rvals[pid] = pool_rvals[pid] + rvals_temp[pid]

        for rv in pool_rvals:
            rvals.extend(rv)
        return rvals
