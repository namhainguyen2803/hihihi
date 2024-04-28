import numpy as np

def calculate_mean_std(data, precision=3):
    mean = np.mean(data)
    std_dev = np.std(data)
    mean_str = "{:.3f}".format(mean)
    std_dev_str = "{:.3f}".format(std_dev)
    return f"{mean_str} \pm {std_dev_str}"

def extract_values(log_path):
    # Initialize lists to store values
    F_list = []
    AD_list = []
    FI_list = []
    ADI_list = []
    RL_values = []

    with open(log_path, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Check if the line contains the desired values
            if '+) Fairness (F):' in line:
                F_list.append(float(line.split(':')[-1].strip()))
            elif '+) Averaging distance (AD):' in line:
                AD_list.append(float(line.split(':')[-1].strip()))
            elif '+) Fairness in images space (FI):' in line:
                FI_list.append(float(line.split(':')[-1].strip()))
            elif '+) Averaging distance in images space (ADI):' in line:
                ADI_list.append(float(line.split(':')[-1].strip()))
            elif 'Reconstruction loss (RL):' in line:
                RL_values.append(float(line.split(':')[-1].strip()))  # Add RL value to the set

    # Check if all RL values are equal
    # assert len(RL_values) == 1, "Reconstruction loss (RL) values are not equal in all datasets"
    return F_list, AD_list, FI_list, ADI_list, RL_values
# Example usage:

if __name__ == "__main__":
    RESULT = ["result3"]
    DATASET = ["mnist"]
    SEED = ["seed_42"]
    LR = ["lr_0.001"]
    FSW = ["fsw_0.1", "fsw_0.5", "fsw_1.0", "fsw_2.0", "fsw_4.0"]
    METHOD = ["EFBSW", "FBSW", "lowerboundFBSW", "OBSW_0.1","OBSW", "OBSW_10.0", "BSW"]
    # METHOD = ["EFBSW"]
    METHOD_NAME = ["es-MFSWB", "us-MFSWB", "s-MFSWB", "MFSWB $\lambda = 0.1$", "MFSWB $\lambda = 1.0$", "MFSWB $\lambda = 10.0$", "USWB"]
    for r in RESULT:
        for d in DATASET:
            for s in SEED:
                for l in LR:
                    res_latex = ""
                    for i in range(len(METHOD)-1, -1, -1):
                        res_method_latex = f"{METHOD_NAME[i]}"
                        for f in FSW:
                            m = METHOD[i]
                            log_path = f"{r}/{d}/{s}/{l}/{f}/{m}/evaluate_{m}.log"
                            F, AD, FI, ADI, RL = extract_values(log_path)

                            res_method_latex += f" & ${calculate_mean_std(F[-3:])}$ & ${calculate_mean_std(AD[-3:])}$ & " + "{:.3f}".format(RL[-1])
                            
                            # print(f"==> {log_path}, {len(RL[:])}, {len(set(RL[:]))}")
                        
                        res_method_latex += "\\\\"
                        print(res_method_latex)
