import os 
import glob

HOME_PATH = os.path.expanduser('~')


main_mapping = {'bonds': 'bondsKernelsGpu', 'black-scholes': 'blackScholesAnalyticEngineKernelsCpu',
        'crc64': 'CRC64', 'gaussian': 'gaussianElim', 'hotspot3D': '3D', 'hmm': 'HiddenMarkovModel',
        'leukocyte': 'detect_main', 'nn': 'nearestNeighbor', 'multimaterial': 'multimat', 
        'su3': 'su3_nn_bench', 'mt': 'MT', 'heartwall': 'kernel', 'halo-finder': 'ForceTreeTest',
        'kmeans': 'cluster', 'saxpy-ompt': 'saxpy', 'dwconv1d': 'timex', 'pingpong': 'main-nccl'}

main_paths = {'convolutionDeformable-cuda': 'src/cuda/dcn_v2_cuda.cu',
              'convolutionDeformable-hip': 'src/hip/dcn_v2_hip.hip',
              'convolutionDeformable-sycl': 'src/sycl/dcn_v2_cuda.cpp', 
              'daphne-hip': 'src/points2image/kernel.cu',
              'eikonal-hip': 'kernel.cu', 
              'heartwall-sycl': 'kernel/kernel.cpp', 
              'miniDGS-cuda': 'src/MaxwellsKernel3d.cu', 
              'miniFE-cuda': 'src/main.cpp', 
              'miniFE-hip': 'src/main.cpp', 
            #   'minmax-hip':, 
            #   'mtf-hip':, 
              'pingpong-sycl': 'main-ccl.cpp', 
              'sph-cuda': 'fluid.cu', 
              'sph-hip': 'fluid.cu', 
              'tpacf-hip': 'compute.cu', 
              'xsbench-cuda': 'Simulation.cu'}


def find_main(kernel_path, kernel): 
    if kernel in main_paths: return [os.path.join(kernel_path, main_paths[kernel])]

    kernel_name = kernel.rsplit('-',1)[0]
    files = glob.glob(os.path.join(kernel_path, 'main.c*'), recursive=False)

    if kernel_name in ['heartwall']:
        files = glob.glob(os.path.join(kernel_path, '**', f'{main_mapping[kernel_name]}.*'), recursive=True)

    if not files:
        files = glob.glob(os.path.join(kernel_path, '**', 'main.c*'), recursive=True)

    if not files:
        
        if kernel_name in main_mapping:
            files = glob.glob(os.path.join(kernel_path, '**', f'{main_mapping[kernel_name]}.*'), recursive=True)
        else:
            files = glob.glob(os.path.join(kernel_path, '**', f'{kernel_name}.*'), recursive=True)

    if not files:
        files = glob.glob(os.path.join(kernel_path, '*.cpp'), recursive=True) + \
                    glob.glob(os.path.join(kernel_path, '*.cu'), recursive=True)

    return files


def get_kernels(dataset_path, detailed=False, err=False):
    kernels = {}
    err_kernels = []
    
    for kernel in os.listdir(dataset_path):
        kernel_path = os.path.join(dataset_path, kernel)
        files = find_main(kernel_path, kernel)
        
        api = kernel.split('-')[-1]
        if len(files) > 1:
            if api in ['hip', 'cuda']:
                files = [path for path in files if path.endswith('.cu')]
            if api in ['omp', 'sycl']:
                files = [path for path in files if not path.endswith('.h') and not path.endswith('.hpp')]
        
        if len(files) > 1:
            files = [path for path in files if 'src' in path[len(kernel_path):].split('/')]

        if len(files) == 1:

            if not detailed:
                kernels[kernel] = files[0]
            else:
                file_path = files[0]
                kernel_name, api = kernel.rsplit('-', 1)

                kernels[kernel] = {'kernel_name': kernel_name,
                                    'parallel_api': api,
                                    'path': file_path
                                    }

        else:
            err_kernels.append(kernel)

    return  (kernels, err_kernels) if err else kernels


    
if __name__=='__main__':
    dataset_path = os.path.join(HOME_PATH, 'HeCBench/src')
    kernels, err_kernels = get_kernels(dataset_path, err=True)

    print(f'Amount of kernels: {len(kernels)}')
    print(err_kernels)