# brka
Resource-constrained parallel machine scheduling with setups, job release/due time, and resource transition limit

Commercial use, redistribution, or adaptation of any part of this repository requires explicit permission from the author. Please contact me for licensing terms or collaboration.

Data instances of UPMSR-PS and IMPR-P benchmarks are available at the following references, respectively:

    1- Lopez-Esteve, A., Perea, F., and Yepes-Borrero, J. C. (2023). Grasp algorithms for the unrelated parallel machines scheduling problem with additional resources during processing and setups. International Journal of Production Research, 61(17):6013–6029.
    2- Caselli, G., Delorme, M., Iori, M., and Magni, C. A. (2024). Exact algorithms for a parallel machine scheduling problem with workforce and contiguity constraints. Computers & Operations Research, 163:106484.


Below are example commands to run the main problem (PMSP), UPMSR-PS, and IPMR-P benchmarks using the BRKGA solver. (See the top of this file for a full description of the PMSP.)
    python grka.py PMSP Instances\PMSP\J180_M4_P3_W3_TW0.5_ME0.8_AP0.txt --solver brkga --threads 1 --max_cpu 360   
    python grka.py UPMSR_PS Instances\UPMSR-PS\100x4_U_50_100_S_100_PU_SU_rep_1.txt --solver brkga --max_cpu 300.0 --threads 1  
    python grka.py IPMR_P Instances\IPMR-P\TEST0-100-4-4-A.txt --solver brkga --threads 1  --max_cpu 200 

To run GRASP on a UPMSR-PS instance:
    python grka.py UPMSR_PS_grasp Instances\UPMSR-PS\100x4_U_50_100_S_100_PU_SU_rep_1.txt --solver random --max_cpu 6.0 --threads 1

Warm-start Option for the UPMSR-PS and IMPR-P benchmarks:
    To enable the warm-start feature, activate the corresponding warm-population function in brkga.py.
