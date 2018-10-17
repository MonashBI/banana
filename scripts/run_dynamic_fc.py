from banana.study.mri.functional.dynamic_fc_to_dump import DynamicFC

d = ('/Volumes/Project/pet/rsfPET/fMRI_analysis/'
     'rsfMRI_dual_regression_ica_80comp/')
good_comp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

dfc = DynamicFC(d, good_comp, ica=False)
dfc.load_subjects()
dfc.windowed_fc(window_tp=45, step=1, sigma=14, method='corr')
dfc.subsampling()
dfc.find_num_clusters_gmm()
dfc.gmm_clustering()
dfc.static_fc()
dfc.save_results(('/Users/francescosforazzini/Desktop/test_dfnc_corr/'), 'all')
