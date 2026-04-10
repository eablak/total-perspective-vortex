## CSP implementation

#### Accordingly csp.py file here is the guidence for how to write your own csp:

<b>Step 1:</b> Covariance Calculation - Covariance Matrix (ft_fit)

<b>Step 2:</b> Eigen Decomposition of a Matrix (ft_fit)

<b>Step 3:</b> Rank eigenvectors with eigenvalues (ft_fit)

After this three steps you have to get your top filters for n_components. Point for the get your components in here is, getting only top filters from all filters. E.g if your X_raw = (120, 64, 641), you have 64 channel and if you give n_components=4 in your csp() after the process you will get only the top 4 filters from 64 filter.

<b>Step 4:</b> Compute Power: compute features -> mean power (ft_fit & ft_transform)

<b>Step 5:</b> Standardize (ft_fit: calculate mean and std & ft_transform: use mean and std)

The aim for step4 and step 5 is converting your raw EEG data for LDA. And LDA can classify with this form. 

<b>Step 6:</b> Transform

Step 6 is using step4 and step5 results for mean and std. The reason for using learned mean and std values is prevent the data leakage.

- Note: Training phrase contains fit and transform but prediction contain only transform.