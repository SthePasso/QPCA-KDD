from qpca import KDD, QPCA
kdd_data = KDD()
models = QPCA()

TOTAL_SAMPLES = kdd_data.X.shape[0]
samples = int((TOTAL_SAMPLES * 1) / 10)   

print("TOTAL_SAMPLES:", TOTAL_SAMPLES)
print("Samples:", samples)

# for j in range(1,6):#, 11): # Size of dataset: 10%, 20%, ..., 100%
#     samples = int((TOTAL_SAMPLES * j) / 10)   
#     df_result_pca = models.main_cpca_correlation(samples = samples, start_at=2, end_at=11)
# print("**********************************************")
for j in range(1,6):#, 11): # Size of dataset: 10%, 20%, ..., 100%
    samples = int((TOTAL_SAMPLES * j) / 100)  
    df_result_qpca = models.main_qpca_q_correlation(samples = samples, start_at=2, end_at=11) #"ibm_brisbane", start_at=2, end_at=20)
print("**********************************************")
# for j in range(1,6):#, 11): # Size of dataset: 10%, 20%, ..., 100%
#     samples = int((TOTAL_SAMPLES * j) / 100)   
#     df_result_qkernel0 = models.main_qpca_rbf_correlation(samples = samples, start_at=2, end_at=11)
# print("**********************************************")
