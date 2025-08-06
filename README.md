# Analytical-Floor-Response-Spectra
Data and application for the journal publication: 

"Analytical floor response spectra for performance-based seismic design of non-structural elements in reinforced concrete frame buildings."  

An application with a graphbical user interface is provided, which performs the analytical methdology proposed in the publication. The application is provided in the following link: 

https://www.dropbox.com/scl/fo/aylva590fd1ta4a5itfcu/AEUnNBoqcUsa3eOss9U5Xaw?rlkey=up6pt5uuashfuic2fqxf7wh9a&st=p06k44y0&dl=0

For the application to work, the executable file must not be moved from the directory where the _internal folder is. Example input ground-motion spectra input files are also provided.  

The python code of the application is provided here and it is named FRSet.py

The application requires as input the acceleration-displacement capacity curve of the supporting structure and a ground-motion response spectrum defined in both acceleration and displacement and for periods from 0 s to 10 s. The ground-motion response spectrum is entered as a csv file with three columns (period, acceleration, and displacement). The first line of the file should contain the headers: T, Sa, Sd in that order. For now, the application only works for supporting structures with uniform mass distribution along their heights and identical inter-storey heights for all storeys. 

The data on the floor response spectra of the archetype RC frames of the study is also provided, as well as the codes used to do the comparisons as Jupyter Notebook codes. Additional data, floor acceleration time histories etc, are available through reasonable request to Roberto J. Merino (r.merinovela@ucl.ac.uk).

If you use any of the codes or data here provided, please cite the following publication:

R.J. Merino, R. Gentile, C. Galasso (2025) "Analytical floor response spectra for performance-based seismic design of non-structural elements in reinforced concrete frame buildings." Under Review. 



