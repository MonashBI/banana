from abc import ABCMeta
from logging import getLogger
from nianalysis.exceptions import (
    NiAnalysisDatasetNameError, NiAnalysisMissingDatasetError)
from nianalysis.pipeline import Pipeline


logger = getLogger('NiAnalysis')


class Study(object):
    """
    Abstract base study class from which all study derive.

    Parameters
    ----------
    name : str
        The name of the study.
    project_id: str
        The ID of the study. For DaRIS it is the project
        id minus the proceeding 1008.2. For XNAT it will be
        the project code. For local archives name of the directory.
    archive : Archive
        An Archive object referring either to a DaRIS, XNAT or local file
        system study
    input_datasets : Dict[str,base.Dataset]
        A dict containing the a mapping between names of study dataset_specs
        and existing datasets (typically acquired from the scanner but can
        also be replacements for generated dataset_specs)
    """

    __metaclass__ = ABCMeta

    def __init__(self, name, project_id, archive, input_datasets):
        self._name = name
        self._project_id = project_id
        self._input_datasets = {}
        # Add each "input dataset" checking to see whether the given
        # dataset_spec name is valid for the study type
        for comp_name, dataset in input_datasets.iteritems():
            if comp_name not in self._dataset_specs:
                raise NiAnalysisDatasetNameError(
                    "Input dataset dataset_spec name '{}' doesn't match any "
                    "dataset_specs in {} studies".format(
                        comp_name, self.__class__.__name__))
            self._input_datasets[comp_name] = dataset
        # Emit a warning if an acquired dataset_spec has not been provided for
        # an "acquired dataset_spec"
        for spec in self.acquired_dataset_specs():
            if spec.name not in self._input_datasets:
                logger.warning(
                    "'{}' acquired dataset_spec was not specified in {} '{}' "
                    "(provided '{}'). Pipelines depending on this dataset "
                    "will not run".format(
                        spec.name, self.__class__.__name__, self.name,
                        "', '".join(self._input_datasets)))
        # TODO: Check that every session has the acquired datasets
        self._archive = archive

    def __repr__(self):
        """String representation of the study"""
        return "{}(name='{}')".format(self.__class__.__name__, self.name)

    def dataset(self, name):
        """
        Returns either the dataset that has been passed to the study __init__
        matching the dataset name provided or the processed dataset that is
        to be generated using the pipeline associated with the generated
        dataset_spec

        Parameters
        ----------
        dataset : Str
            Name of the dataset_spec to the find the corresponding acquired
            dataset or processed dataset to be generated
        """
        try:
            dataset = self._input_datasets[name]
        except KeyError:
            try:
                dataset = self._dataset_specs[name].apply_prefix(self.name +
                                                                 '_')
            except KeyError:
                raise NiAnalysisDatasetNameError(
                    "'{}' is not a recognised dataset_spec name for {} studies."
                    .format(name, self.__class__.__name__))
            if not dataset.processed:
                raise NiAnalysisMissingDatasetError(
                    "Acquired (i.e. non-generated) dataset '{}' is required "
                    "for requested pipelines but was not supplied when the "
                    "study was initiated.".format(name))
        return dataset

    @property
    def project_id(self):
        """Accessor for the project id"""
        return self._project_id

    @property
    def name(self):
        """Accessor for the unique study name"""
        return self._name

    @property
    def archive(self):
        """Accessor for the archive member (e.g. Daris, XNAT, MyTardis)"""
        return self._archive

    def _create_pipeline(self, *args, **kwargs):
        """
        Creates a Pipeline object, passing the study (self) as the first
        argument
        """
        return Pipeline(self, *args, **kwargs)

    @classmethod
    def dataset_spec(cls, name):
        """
        Return the dataset_spec, i.e. the template of the dataset expected to be
        supplied or generated corresponding to the dataset_spec name.

        Parameters
        ----------
        name : Str
            Name of the dataset_spec to return
        """
        return cls._dataset_specs[name]

    @classmethod
    def dataset_spec_names(cls):
        """Lists the names of all dataset_specs defined in the study"""
        return cls._dataset_specs.iterkeys()

    @classmethod
    def dataset_specs(cls):
        """Lists all dataset_specs defined in the study class"""
        return cls._dataset_specs.itervalues()

    @classmethod
    def acquired_dataset_specs(cls):
        """
        Lists all dataset_specs defined in the study class that are provided as
        inputs to the study
        """
        return (c for c in cls.dataset_specs() if not c.processed)

    @classmethod
    def generated_dataset_specs(cls):
        """
        Lists all dataset_specs defined in the study class that are typically
        generated from other dataset_specs (but can be overridden in input
        datasets)
        """
        return (c for c in cls.dataset_specs() if c.processed)

    @classmethod
    def generated_dataset_spec_names(cls):
        """Lists the names of generated dataset_specs defined in the study"""
        return (c.name for c in cls.generated_dataset_specs())

    @classmethod
    def acquired_dataset_spec_names(cls):
        """Lists the names of acquired dataset_specs defined in the study"""
        return (c.name for c in cls.acquired_dataset_specs())


class MultiStudy(Study):
    """
    Abstract base class for all studies that combine multiple studies into a
    a combined study
    """

    __metaclass__ = ABCMeta

    # NB: Subclasses are expected to have a class member named
    #     component_classes that defines the components that make up the multi-
    #     component study and the mapping of their dataset names

    def __init__(self, name, project_id, archive, input_datasets):
        super(MultiStudy, self).__init__(name, project_id, archive,
                                         input_datasets)
        self._sub_components = {}
        for comp_name, (cls, dataset_map) in self.component_specs.iteritems():
            # Create copies of the input datasets to pass to the __init__
            # method of the generated components
            mapped_inputs = []
            for dataset in input_datasets:
                try:
                    mapped_inputs.append(
                        dataset.renamed_copy(dataset_map[dataset.name]))
                except KeyError:
                    pass
            # Create sub-component
            sub_component = cls(name + '_' + comp_name, project_id, archive,
                                mapped_inputs)
            # Set component as attribute
            setattr(self, comp_name, sub_component)
            # Append to dictionary of sub_components
            assert comp_name not in self._sub_components, (
                "duplicate component names '{}'".format(comp_name))
            self._sub_components[comp_name] = sub_component

    @property
    def sub_components(self):
        return self._sub_components.itervalues()

    @classmethod
    def translate_getter(cls, comp_name, pipeline_getter):
        """
        A "decorator" (although not intended to be used with @) for translating
        pipeline getter methods from a sub-component of a MultiStudy. Returns
        a new method that calls the getter on the specified sub-component then
        translates the pipeline to the MultiStudy.
        """
        def translated(self, **kwargs):
            return pipeline_getter(self._sub_components[comp_name],
                                   **kwargs).translate(self)
        return translated


def set_dataset_specs(*comps, **kwargs):
    dct = {}
    for comp in comps:
        if comp.name in dct:
            assert False, ("Multiple values for '{}' found in component list"
                           .format(comp.name))
        dct[comp.name] = comp
    if 'inherit_from' in kwargs:
        combined = set_dataset_specs(*set(kwargs['inherit_from']))
        # Allow the current components to override the inherited ones
        combined.update(dct)
        dct = combined
    return dct
