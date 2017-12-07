## Bootstrap script for setting up a valid Python environment.
## It is currently customized for Microsoft Azure Notebooks Library.

install_dir="$HOME/library"
repo_url="https://github.com/rballester/ttrecipes.git"
repo_branch="master"

cd ${install_dir}
if [ ! -d ${install_dir}/ttrecipes ]; then
    git clone -b ${repo_branch} ${repo_url}
else
    git checkout examples/sensitivity_analysis/Sensitivity\ Analysis\ Examples.ipynb && git pull &
fi

#echo "Installing Python 3 packages..."
#pip3 install --user six numpy cython
#pip3 install --user -r requirements.txt

#echo "Installing Python 2 packages..."
#pip2 install --user six numpy cython
#pip2 install --user -r requirements.txt

