# Seismic Modeling with Devito  

## Repository Overview  
This repository contains Python-based seismic modeling workflows utilizing the **Devito** framework for high-performance finite difference computations. The project focuses on wave propagation simulations, reverse time migration (RTM), and full waveform inversion (FWI) in seismic exploration.  

Due to confidentiality agreements, certain datasets and modeling results are not publicly available. For access to specific outputs or collaboration inquiries, please contact the repository owner.  

🔗 **Repository Link:** [https://github.com/spaceswimmer/Seismic_modeling](https://github.com/spaceswimmer/Seismic_modeling)  

## System Requirements  
- **Operating System:** Unix-based (Linux/macOS)  
- **GPU Acceleration:** NVIDIA CUDA Toolkit (v11.0+)  
- **Python Dependencies:**  
  - Devito (v4.8+)  
  - NumPy, SciPy, Matplotlib
  (using requirements.txt prefered) 

## Repository Structure  
```  
Seismic_modeling/  
├── /data/                  # data to run python scripts
├── /docs/                  # Technical notes and theory references
├── /results/               # Results of scripts  
├── /src/                   # Core Python scripts  
│   ├── 2d_acoustic             # 2d Acoustic seismic modeling scripts
│   ├── 2d_elastic              # 2d Elastic seismic modeling scripts
│   ├── 2d_vankor               # 2d Elastic seismic modeling scripts on vankor data
│   ├── 2d_vibro                # 2d Elastic seismic modeling scripts for vibrator source
│   ├── 2d_VSP_RTM              # 2d Reverse time migration scripts
│   ├── 3d acoustic             # 3d Elastic seismic modeling scripts
│   ├── Experimental code       # VIP code
│   └── scratch/                # utility library with custom solvers and functions
├── /tests/                 # Unit and integration tests (pytest) (VIP not implemented yet) 
├── /requiremets_conda.txt  # Conda env requirements
├── /requiremets.txt        # Python venv requirements
└── README.md               # Project documentation  
```  

## Key Features  
- **Optimized PDE Solvers:** Leverages Devito's symbolic computation for wave equation discretization.  
- **HPC-Ready:** Supports multi-GPU parallelism via CUDA and MPI.  
- **Reproducible Workflows:** Configuration-driven execution for consistent benchmarking.  

## Installation  
1. **Set up CUDA Toolkit:**  
    Install CUDA Toolkit from official NVIDIA website:
    🔗 **CUDA Toolkit:** [https://developer.nvidia.com/cuda-downloads?target_os=Linux](CUDA Toolkit)  
2. **Create Conda Environment or Python virtual environment:**  
    Conda approach:
    ```bash  
    conda create -n devito --file requirements_conda.txt  
    conda activate devito   
    ```
    Python venv approach:
    ```bash
    pip install virtualenv
    virtualenv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage  
Execute simulations with predefined configs:  
```bash  
python src/2d_elastic/AM_model.py 250 0 0.02 "../data/AM_model/recorders.txt" -r "../results/2d_elastic" -gpu   
```

## License  
Proprietary code. Unauthorized redistribution prohibited.  

For collaboration or data access requests, please open a GitHub Issue or contact the maintainer.  

---  
*Note: Computational results and proprietary velocity models are omitted per client agreements.*
