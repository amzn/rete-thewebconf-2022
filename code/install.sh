cd Model/para_samplers/
rm -rf pybind11
git clone https://github.com/pybind/pybind11.git
cd ..
conda install -c anaconda cmake
conda install -c conda-forge ninja
pip install pybind11
cd ..
pip install ./para_samplers
