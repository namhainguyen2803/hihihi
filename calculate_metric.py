import numpy as np
import json

def calculate_mean_std(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    # mean_format = "{:.{}f}".format(mean, precision)
    # std_dev_format = "{:.{}f}".format(std_dev, precision)
    return mean, std_dev


def extract_values(log_path):
    # Initialize lists to store values
    WG_list = []
    LP_list = []
    FRL_list = []
    WRL_list = []
    RL_values = []
    F_latent_list = []
    W_latent_list = []
    F_image_list = []
    W_image_list = []

    with open(log_path, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Check if the line contains the desired values
            
            if '+) Wasserstein distance between generated and real images (WG):' in line:
                WG_list.append(float(line.split(':')[-1].strip()))
            elif '+) Wasserstein distance between posterior and prior distribution (LP)' in line:
                LP_list.append(float(line.split(':')[-1].strip()))
                
            elif '+) Fairness of Reconstruction Loss (FRL):' in line:
                FRL_list.append(float(line.split(':')[-1].strip()))
            elif '+) Averaging distance of Reconstruction Loss (WRL):' in line:
                WRL_list.append(float(line.split(':')[-1].strip()))
                
            elif '+) Reconstruction loss (RL):' in line:
                RL_values.append(float(line.split(':')[-1].strip()))
                
            elif '+) Fairness (F):' in line:
                F_latent_list.append(float(line.split(':')[-1].strip()))
            elif '+) Averaging distance (W):' in line:
                W_latent_list.append(float(line.split(':')[-1].strip()))
                
            elif '+) Fairness in images space (FI):' in line:
                F_image_list.append(float(line.split(':')[-1].strip()))
            elif '+) Averaging distance in images space (WI):' in line:
                W_image_list.append(float(line.split(':')[-1].strip()))

    return WG_list, LP_list, FRL_list, WRL_list, RL_values, F_latent_list, W_latent_list, F_image_list, W_image_list
# Example usage:

if __name__ == "__main__":
    
    json_file_path = "big_dick_data.json"

    with open(json_file_path, "r") as json_file:
        big_dick = json.load(json_file)
    
    # list_FSW_value = ["0.1", "0.5", "1.0", "2.0", "4.0"]
    list_FSW_value = ["0.0"]
    CKP = "100"
    
    all_latex_codes = ""
    template_file_path = "table_template.txt"
    
    for FSW_value in list_FSW_value:
    
        RESULT = ["result3"]
        DATASET = ["mnist"]
        SEED = ["seed_42"]
        LR = ["lr_0.001"]
        FSW = [f"fsw_{FSW_value}"]
        METHOD = ["EFBSW", "FBSW", "lowerboundFBSW", "OBSW_0.1","OBSW", "OBSW_10.0", "BSW"]
        METHOD_NAME = ["es-MFSWB", "us-MFSWB", "s-MFSWB", "MFSWB $\lambda = 0.1$", "MFSWB $\lambda = 1.0$", "MFSWB $\lambda = 10.0$", "USWB"]
        # METHOD = ["None"]
        # METHOD_NAME = ["None"]
        
        num_data = 100000
        
        for r in RESULT:
            for d in DATASET:
                for s in SEED:
                    for l in LR:
                        res_latex = ""
                        for i in range(len(METHOD)-1, -1, -1):
                            res_method_latex = f"{METHOD_NAME[i]}"
                            for f in FSW:
                                m = METHOD[i]
                                log_path = f"{r}/{d}/{s}/{l}/{f}/{m}/evaluate_epoch_{CKP}_{m}_2.log"
                                WG_list, LP_list, FRL_list, WRL_list, RL_values, F_latent_list, W_latent_list, F_image_list, W_image_list = extract_values(log_path)
                                assert len(set(RL_values)) == 1               
                                print(f"==> {log_path}, {len(RL_values[:])}, {len(set(RL_values[:]))}")
                                
                                num_data = min(num_data, len(RL_values[:]))
                        
        print(num_data)
        
        num_data = 10
        
        DATA_value = ""
        for r in RESULT:
            for d in DATASET:
                for s in SEED:
                    for l in LR:
                        res_latex = ""
                        for i in range(len(METHOD)-1, -1, -1):
                            res_method_latex = f"{METHOD_NAME[i]}"
                            for f in FSW:
                                m = METHOD[i]
                                log_path = f"{r}/{d}/{s}/{l}/{f}/{m}/evaluate_epoch_{CKP}_{m}_2.log"
                                WG_list, LP_list, FRL_list, WRL_list, RL_values, F_latent_list, W_latent_list, F_image_list, W_image_list = extract_values(log_path)

                                mean_LP, std_LP = calculate_mean_std(LP_list[-num_data:])
                                mean_LP *= 10**3
                                std_LP *= 10**3
                                mean_WG, std_WG = calculate_mean_std(WG_list[-num_data:])
                                
                                mean_F_latent, std_F_latent = calculate_mean_std(F_latent_list[-num_data:])
                                mean_W_latent, std_W_latent = calculate_mean_std(W_latent_list[-num_data:])
                                mean_F_image, std_F_image = calculate_mean_std(F_image_list[-num_data:])
                                mean_W_image, std_W_image = calculate_mean_std(W_image_list[-num_data:])

                                assert len(set(RL_values)) == 1
                                print(F_latent_list[-num_data:])
                                res_method_latex += " & {:.3f}".format(RL_values[-2]) + "& {:.3f}".format(FRL_list[-2]) + "& {:.3f}".format(mean_LP) + "& {:.3f}".format(mean_F_latent) + "& {:.3f}".format(mean_W_latent) + " & {:.3f}".format(mean_WG) + "& {:.3f}".format(mean_F_image) + "& {:.3f}".format(mean_W_image)           
                            
                            res_method_latex += "\\\\"
                            res_method_latex += "\n"
                            DATA_value += res_method_latex

        with open(template_file_path, "r") as template_file:
            latex_template = template_file.read()
            
            latex_code = latex_template.replace("@FSW", FSW_value)
            latex_code = latex_code.replace("@EPOCHS", CKP)
            latex_code = latex_code.replace("@DATA", DATA_value)
        
        all_latex_codes += latex_code + "\n"

        output_file_path = "output.txt"
        with open(output_file_path, "w") as output_file:
            output_file.write(all_latex_codes)

        print("Updated LaTeX code has been written to:", output_file_path)