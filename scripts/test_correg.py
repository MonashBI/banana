import nipype.interfaces.spm as spm
coreg = spm.Coregister()
coreg.inputs.source = '/Users/tclose/git/mbi/arcana/test/_data/t1.nii.gz'
coreg.inputs.target = '/Users/tclose/git/mbi/arcana/test/_data/t2.nii.gz'
coreg.run()
print 'done'
