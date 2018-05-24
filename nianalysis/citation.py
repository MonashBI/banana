from arcana.citation import Citation


mrtrix_cite = Citation(
    short_name="mrtrix",
    authors=["Tournier, J-D"],
    title="MRtrix Package",
    institute="Brain Research Institute, Melbourne, Australia",
    url="https://github.com/MRtrix3/mrtrix3",
    year=2012)

matlab_cite = Citation(
    short_name="matlab",
    authors=["A"],
    title="MATLAB",
    institute="Mathworks",
    url="https://placeholder.com",
    year=2014)

sti_cites = Citation(
    short_name="sti",
    authors=["A"],
    title="Placeholder",
    institute="Placeholder",
    url="https://placeholder.com",
    year=2014)

eddy_cite = Citation(
    short_name='Eddy',
    authors=["Andersson, J. L.", "Sotiropoulos, S. N."],
    title=(
        "An integrated approach to correction for "
        "off-resonance effects and subject movement in "
        "diffusion MR imaging"),
    journal="NeuroImage", year=2015, volume=125,
    pages=(1063, 1078)),

fsl_cite = Citation(
    short_name="FSL",
    authors=["Smith, S. M.", "Jenkinson, M.",
             "Woolrich, M. W.", "Beckmann, C. F.",
             "Behrens, T. E.", "Johansen- Berg, H.",
             "Bannister, P. R.", "De Luca, M.", "Drobnjak, I.",
             "Flitney, D. E.", "Niazy, R. K.", "Saunders, J.",
             "Vickers, J.", "Zhang, Y.", "De Stefano, N.",
             "Brady, J. M. & Matthews, P. M."],
    title=(
        "Advances in functional and structural MR image "
        "analysis and implementation as FSL"),
    journal="NeuroImage", year=2004, volume=23,
    pages=("S208", "S219"))

distort_correct_cite = Citation(
    short_name="skare_distort_correct",
    authors=["Skare, S.", "Bammer, R."],
    title=(
        "Jacobian weighting of distortion corrected EPI data"),
    journal=(
        "Proceedings of the International Society for Magnetic"
        " Resonance in Medicine"), year=2010, pages=(5063,))

topup_cite = Citation(
    short_name="Topup",
    authors=["Andersson, J. L.", "Skare, S. & Ashburner, J."],
    title=(
        "How to correct susceptibility distortions in "
        "spin-echo echo-planar images: application to "
        "diffusion tensor imaging"),
    journal="NeuroImage", year=2003, volume=20,
    pages=(870, 888))

noddi_cite = Citation(
    short_name="noddi",
    authors=["Zhang, H.", "Schneider T.", "Wheeler-Kingshott C. A.",
             "Alexander, D. C."],
    title=(
        "NODDI: Practical in vivo neurite orientation dispersion and density "
        "imaging of the human brain"),
    journal='NeuroImage', year=2012, volume=61, pages=(1000, 1016))

bet_cite = Citation(
    short_name="BET",
    authors=['Smith, S. M.'],
    title=("Fast robust automated brain extraction"),
    journal="Human Brain Mapping",
    volume=17, issue=3, pages=(143, 155), month='November', year=2002)

bet2_cite = Citation(
    short_name="BET2",
    authors=["Jenkinson, M.", "Pechaud, M.", 'Smith, S. M.'],
    title=("MR-based estimation of brain, skull and scalp surfaces"),
    proceedings=(
        "Eleventh Annual Meeting of the Organization for Human Brain Mapping"),
    year=2005)

fast_cite = Citation(
    short_name="FAST",
    authors=["Zhang, Y.", "Brady, M.", "Smith, S."],
    title=("Segmentation of brain MR images through a hidden Markov random "
           "field model and the expectation-maximization algorithm"),
    proceedings=("IEEE Transactions on Medical Imaging"),
    year=2001, volume=20, pages='45-57')

n4_cite = Citation(
    short_name='n4',
    authors=["Tustison, N.", " Avants, B.", " Cook, P.", " Zheng, Y.",
             " Egan, A.", " Yushkevich, P.", "Gee, J."],
    title=("N4ITK: Improved N3 Bias Correction"),
    proceedings=("IEEE Transactions on Medical Imaging"), year=2010, volume=29,
    pages='1310-1320')


tbss_cite = Citation(
    short_name='tbss',
    authors=['Smith, S.M.', 'Jenkinson, M.', 'Johansen-Berg, H.',
             'Rueckert, D.', 'Nichols, T.E.', 'Mackay, C.E.', 'Watkins, K.E.',
             'Ciccarelli, O.', 'Cader, M.Z.', 'Matthews, P.M.',
             'Behrens, T.E.J.'],
    title=("Tract-based spatial statistics: Voxelwise analysis of "
           "multi-subject diffusion data"),
    journal="NeuroImage", volume=31, pages='1487-1505', year=2006)

spm_cite = Citation(
    short_name='spm',
    authors=['Penny, W.', 'Friston, K.', 'Ashburner, J.', 'Kiebel, S.',
             'Nichols, T.'],
    title=("Statistical Parametric Mapping: The Analysis of Functional Brain "
           "Images"),
    year=2006)

optimal_t1_bet_params_cite = Citation(
    short_name='optimal_t1_bet_params',
    authors=[
        'Popescu V.', 'Battaglini M.', 'Hoogstrate W.S.', 'Verfaillie S.C.',
        'Sluimer I.C.', 'van Schijndel R.A.', 'van Dijk B.W.', 'Cover K.S.',
        'Knol D.L.', 'Jenkinson M.', 'Barkhof F.', 'de Stefano N.',
        'Vrenken H.'],
    title=("Optimizing parameter choice for FSL-Brain Extraction Tool (BET) "
           "on 3D T1 images in multiple sclerosis."),
    journal='Neuroimage', volume=61, year=2012, issue=4)

dwidenoise_cites = [
    Citation(
        short_name='dewidenoise1',
        authors=['Veraart, J.', 'Fieremans. E.', 'Novikov, D.S.'],
        title='Diffusion MRI noise mapping using random matrix theory',
        journal='Magn. Res. Med.', volume=76, issue=5, pages='1582-1593',
        year=2016, doi='10.1002/mrm.26059'),
    Citation(
        short_name='dewidenoise2',
        authors=['Veraart, J.', 'Novikov, D.S.', 'Christiaens, D.',
                 'Adesaron, B.', 'Sijbers, J.', 'Fieremans. E.'],
        title='Denoising of diffusion MRI using random matrix theory',
        journal='NeuroImage', volume=142, pages='394-406',
        year=2016, doi='10.1016/j.neuroimage.2016.08.016')]

freesurfer_cites = [
    Citation(
        short_name='freesurfer1',
        authors=['Dale, A.M.', 'Fischl, B.', 'Sereno, M.I.'],
        title=("Cortical surface-based analysis. I. Segmentation and surface "
               "reconstruction"),
        journal='Neuroimage', volume=9,
        pages='179-194', year=1999),
    Citation(
        short_name='freesurfer2',
        authors=['Dale, A.M.', 'Sereno, M.I.'],
        title=("Improved localization of cortical activity by combining EEG "
               "and MEG with MRI cortical surface reconstruction: a linear "
               "approach"),
        journal='J Cogn Neurosci', volume=5,
        pages='162-176', year=1993),
    Citation(
        short_name='freesurfer3',
        authors=['Desikan, R.S.', 'Segonne, F.', 'Fischl, B.', 'Quinn, B.T.',
                 'Dickerson, B.C.', 'Blacker, D.', 'Buckner, R.L.',
                 'Dale, A.M.', 'Maguire, R.P.', 'Hyman, B.T.', 'Albert, M.S.',
                 'Killiany, R.J.'],
        title=("An automated labeling system for subdividing the human "
               "cerebral cortex on MRI scans into gyral based regions "
               "of interest"),
        journal='Neuroimage', volume=31,
        pages='968-980', year=2006),
    Citation(
        short_name='freesurfer4',
        authors=['Fischl, B.', 'Dale, A.M.'],
        title=("Measuring the thickness of the human cerebral cortex from "
               "magnetic resonance images"),
        journal='Proc Natl Acad Sci USA', volume=97,
        pages='11050-11055', year=2000),
    Citation(
        short_name='freesurfer5',
        authors=['Fischl, B.', 'Liu, A.', 'Dale, A.M.'],
        title=("Automated manifold surgery: constructing geometrically "
               "accurate and topologically correct models of the human "
               "cerebral cortex"),
        journal='IEEE Trans Med Imaging', volume=20,
        pages='70-80', year=2001),
    Citation(
        short_name='freesurfer6',
        authors=['Fischl, B.', 'Salat, D.H.', 'Busa, E.', 'Albert, M.',
                 'Dieterich, M.', 'Haselgrove, C.', 'van der Kouwe, A.',
                 'Killiany, R.', 'Kennedy, D.', 'Klaveness, S.',
                 'Montillo, A.', 'Makris, N.', 'Rosen, B.', 'Dale, A.M.'],
        title=("Whole brain segmentation: automated labeling of "
               "neuroanatomical structures in the human brain"),
        journal='Neuron', volume=33,
        pages='341-355', year=2002),
    Citation(
        short_name='freesurfer7',
        authors=['Fischl, B.', 'Salat, D.H.', 'van der Kouwe, A.J.',
                 'Makris, N.', 'Segonne, F.', 'Quinn, B.T.', 'Dale, A.M.'],
        title=("Sequence-independent segmentation of magnetic resonance "
               "images"),
        journal='Neuroimage 23 Suppl', volume=1,
        pages='S69-84', year=2004),
    Citation(
        short_name='freesurfer8',
        authors=['Fischl, B.', 'Sereno, M.I.', 'Dale, A.M.'],
        title=("Cortical surface-based analysis. II: Inflation, flattening, "
               "and a surface-based coordinate system"),
        journal='Neuroimage', volume=9,
        pages='195-207', year=1999),
    Citation(
        short_name='freesurfer9',
        authors=['Fischl, B.', 'Sereno, M.I.', 'Tootell, R.B.', 'Dale, A.M.'],
        title=("High-resolution intersubject averaging and a coordinate "
               "system for the cortical surface"),
        journal='Hum Brain Mapp', volume=8,
        pages='272-284', year=1999),
    Citation(
        short_name='freesurfer10',
        authors=['Fischl, B.', 'van der Kouwe, A.', 'Destrieux, C.',
                 'Halgren, E.', 'Segonne, F.', 'Salat, D.H.', 'Busa, E.',
                 'Seidman, L.J.', 'Goldstein, J.', 'Kennedy, D.',
                 'Caviness, V.', 'Makris, N.', 'Rosen, B.', 'Dale, A.M.'],
        title=("Automatically parcellating the human cerebral cortex"),
        journal='Cereb Cortex', volume=14,
        pages='11-22', year=2004),
    Citation(
        short_name='freesurfer11',
        authors=['Han, X.', 'Jovicich, J.', 'Salat, D.', 'van der Kouwe, A.',
                 'Quinn, B.', 'Czanner, S.', 'Busa, E.', 'Pacheco, J.',
                 'Albert, M.', 'Killiany, R.', 'Maguire, P.', 'Rosas, D.',
                 'Makris, N.', 'Dale, A.', 'Dickerson, B.', 'Fischl, B.'],
        title=("Reliability of MRI-derived measurements of human cerebral "
               "cortical thickness: the effects of field strength, scanner "
               "upgrade and manufacturer"),
        journal='Neuroimage', volume=32,
        pages='180-194', year=2006),
    Citation(
        short_name='freesurfer12',
        authors=['Jovicich, J.', 'Czanner, S.', 'Greve, D.', 'Haley, E.',
                 'van der Kouwe, A.', 'Gollub, R.', 'Kennedy, D.',
                 'Schmitt, F.', 'Brown, G.', 'Macfall, J.', 'Fischl, B.',
                 'Dale, A.'],
        title=("Reliability in multi-site structural MRI studies: effects of "
               "gradient non-linearity correction on phantom and human data"),
        journal='Neuroimage', volume=30,
        pages='436-443', year=2006),
    Citation(
        short_name='freesurfer13',
        authors=['Kuperberg, G.R.', 'Broome, M.R.', 'McGuire, P.K.',
                 'David, A.S.', 'Eddy, M.', 'Ozawa, F.', 'Goff, D.',
                 'West, W.C.', 'Williams, S.C.', 'van der Kouwe, A.J.',
                 'Salat, D.H.', 'Dale, A.M.', 'Fischl, B.'],
        title=("Regionally localized thinning of the cerebral cortex in "
               "schizophrenia"),
        journal='Arch Gen Psychiatry', volume=60,
        pages='878-888', year=2003),
    Citation(
        short_name='freesurfer14',
        authors=['Reuter, M.', 'Schmansky, N.J.', 'Rosas, H.D.', 'Fischl, B.'],
        title=("Within-Subject Template Estimation for Unbiased Longitudinal "
               "Image Analysis"),
        journal='Neuroimage', volume=61, issue=4,
        pages='1402-1418', year=2012,
        pdf='http://reuter.mit.edu/papers/reuter-long12.pdf'),
    Citation(
        short_name='freesurfer15',
        authors=['Reuter, M.', 'Fischl, B.'],
        title=("Avoiding asymmetry-induced bias in longitudinal image "
               "processing"),
        journal='Neuroimage', volume=57, issue=1,
        pages='19-21', year=2011,
        pdf='http://reuter.mit.edu/papers/reuter-bias11.pdf'),
    Citation(
        short_name='freesurfer16',
        authors=['Reuter, M.', 'Rosas, H.D.', 'Fischl, B.'],
        title=("Highly Accurate Inverse Consistent Registration: A Robust "
               "Approach"),
        journal='Neuroimage', volume=53, issue=4,
        pages='1181-1196', year=2010,
        pdf='http://reuter.mit.edu/papers/reuter-robreg10.pdf'),
    Citation(
        short_name='freesurfer17',
        authors=['Rosas, H.D.', 'Liu, A.K.', 'Hersch, S.', 'Glessner, M.',
                 ' Ferrante, R.J.', 'Salat, D.H.', 'van der Kouwe, A.',
                 ' Jenkins, B.G.', 'Dale, A.M.', 'Fischl, B.'],
        title=("Regional and progressive thinning of the cortical ribbon in "
               "Huntington's disease"),
        journal='Neurology', volume=58,
        pages='695-701', year=2002),
    Citation(
        short_name='freesurfer18',
        authors=['Salat, D.H.', 'Buckner, R.L.', 'Snyder, A.Z.',
                 ' Greve, D.N.', 'Desikan, R.S.', 'Busa, E.', 'Morris, J.C.',
                 'Dale, A.M.', 'Fischl, B.'],
        title=("Thinning of the cerebral cortex in aging"),
        journal='Cereb Cortex', volume=14,
        pages='721-730', year=2004),
    Citation(
        short_name='freesurfer19',
        authors=['Segonne, F.', 'Dale, A.M.', 'Busa, E.', 'Glessner, M.',
                 'Salat, D.', 'Hahn, H.K.', 'Fischl, B.'],
        title=("A hybrid approach to the skull stripping problem in MRI"),
        journal='Neuroimage', volume=22,
        pages='1060-1075', year=2004),
    Citation(
        short_name='freesurfer20',
        authors=['Segonne, F.', 'Pacheco, J.', 'Fischl, B.'],
        title=("Geometrically accurate topology-correction of cortical "
               "surfaces using nonseparating loops"),
        journal='IEEE Trans Med Imaging', volume=26,
        pages='518-529', year=2007),
    Citation(
        short_name='freesurfer21',
        authors=['Sled, J.G.', 'Zijdenbos, A.P.', 'Evans, A.C.'],
        title=("A nonparametric method for automatic correction of intensity "
               "nonuniformity in MRI data"),
        journal='IEEE Trans Med Imaging', volume=17,
        pages='87-97', year=1998)]
