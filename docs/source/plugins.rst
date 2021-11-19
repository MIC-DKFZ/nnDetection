Plugins
=======
nnDetection Plugins are a conventient way to integrate external components into nnDetection without rewriting the internal code base.
By registering custom modules to the registries (see Developer Guide for more info on that) the default behavior can be customized resulting in complete new detection pipelines.

Setting up a new Plugin
-----------------------
An empty template plugin can be found in `plugins/nndetection-template`.

Setting up a new plugin can be done with some simple steps:

1. **Copy the template folder into a new repository** (if you are Git otherwise, otherwise any other folder suffices)
2. Customize `setup.py` (scroll to the bottom and change the following entries in the `setup` function)
    - `name`: give your plugin a name (this name will also be used for imports of your python package) 
    - `author`: change name
    - `maintainer_email`: change email
    - `version`: (optionally) customize the version of your plugin
3. **Add additional requirements to `requirements.txt`** (these will be automatically installed when someone installs the plugin)
4. **Add a license file for the plugin**
5. **Change `nndet_project` to the name which was used in the `setup.py` file**
6. Replace `[project]` inside the hydra plugin
    - rename folder inside `hydra_plugins`
    - rename python file inside `hydra_plugins/[project]-searchpath_plugin`
    - rename variables inside `hydra_plugins/[project]-searchpath_plugin/[project]_searchpath_plugin.py`
7. **Customize `README.md`**

Working with a Plugin
---------------------
After the plugin is configured correctly it can be installed as a usual python package (e.g. via pip).
Custom Config Files will be automatically added to the search path of hydra while custom components need to be registered manually.
Please refer to the section below for more information:

Custom Config Files
*******************
Plugins can add new config files by creating them inside the `nndet_[project]/conf` folder.
The inner folder structure needs to follow them same naming as the native nnDetection repository.

Custom Modules
**************
By leveraging the Registries of nnDetection new modules can be written and integrated without rewriting nnDetection code.
Unfortunately, the registries will only register modules which are imported during runtime and thus it is necesary to tell nnDetection which files to import.
The simplest way to achieve this, is to create a custom config file with the entry `additional_imports: ["nndet_[project]"]` to it.
To automatically include all custom modules from the plugins they should be imported by the root `__init__.py` of your plugin.
Alternatively, it is also possible to add each file via a separate item or provide the overwrite via the command line.

Official Plugins
----------------
#TODO
