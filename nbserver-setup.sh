## Bootstrap script for setting up a valid Python environment.
## It is currently customized for Microsoft Azure Notebooks Library.

install_dir="$HOME/library"
repo_name="ttrecipes"
repo_url="https://github.com/rballester/ttrecipes.git"
repo_branch="master"

if [ ! -d ${install_dir}/${repo_name} ]; then
    cd ${install_dir} && git clone -b ${repo_branch} ${repo_url}
else
    cd ${install_dir}/${repo_name} && git fetch && git checkout ${repo_branch} && git pull
fi

cd ${install_dir}/${repo_name}

echo "Installing Python 3 packages..."
pip3 install --user six numpy cython
pip3 install --user -r requirements.txt

#pip3 install --user six cython
#pip3 install --user git+https://github.com/oseledets/ttpy@4e4cf970fad9a4ad39efcab3d009326a71113a49#egg=ttpy
#pip3 install --user -e .
