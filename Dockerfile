FROM jupyter/tensorflow-notebook
WORKDIR /home/jovyan/work
RUN pip install --upgrade pip
RUN pip install PyQt5 'widgetsnbextension==3.1.*'
RUN pip install --upgrade tensorflow keras pandas scikit-image scikit-learn scipy matplotlib
ADD . /home/jovyan/work
CMD ["start-notebook.sh"]