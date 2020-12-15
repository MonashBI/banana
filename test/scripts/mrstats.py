from banana.interfaces.mrtrix.utils import MRStats

stats = MRStats(allvolumes=True)

stats.inputs.in_file = '/Users/tclose/Data/residual-output/ABC_0003_IG_LTFU2/ss3t.nii.gz'

result = stats.run()

print(result.outputs)
