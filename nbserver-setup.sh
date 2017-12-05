## Bootstrap script for setting up a valid Python environment.
## It is currently customized for Microsoft Azure Notebooks Library.

install_dir="$HOME/library"
repo_url="https://github.com/rballester/ttrecipes.git"
repo_branch="master"

cd ${install_dir}
if [ ! -d ${install_dir}/ttrecipes ]; then
    git clone -b ${repo_branch} ${repo_url}
else
    git fetch && git checkout ${repo_branch} && git checkout . && git pull
fi

echo "Installing Python 3 packages..."
pip3 install --user six numpy cython
pip3 install --user -r requirements.txt

#pip3 install --user six cython
#pip3 install --user git+https://github.com/oseledets/ttpy@4e4cf970fad9a4ad39efcab3d009326a71113a49#egg=ttpy
#pip3 install --user -e .
