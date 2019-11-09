In Python 
=========

An application of the DWI analysis to a dataset stored in a local directory

.. code-block:: python

    from banana import BasicRepo, StaticEnv, SingleProc
    from banana.analysis.mri import DwiAnalysis
    from banana.file_format import dicom_format

    # Initialise analysis, selecting data corresponding to the data
    # specified in the local directory repository and parameters
    # used in the processing
    your_analysis = DwiAnalysis(
        name='your_analysis',
        repository=BasicRepo('/path/to/local/archive'),
        processor=SingleProc('/my/work/dir'),
        environment=StaticEnv(),
        inputs=[
            InputFilesets('series', 'RL-epi-diffusion-series', dicom_format),
            InputFilesets('reverse_phase', 'LR-epi-diffusion-ref',
                          dicom_format)],
        parameters={'num_global_tracks': 1e12,
                    'global_tracks_cutoff': 0.01})

    # Execute the pipelines required to generate file 5 and field 2
    # and return handle to generated data
    fa, global_tracks = analysis.data(['fa', 'global_tracks'])
