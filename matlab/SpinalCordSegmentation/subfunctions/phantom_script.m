dims        = [70 70 300 64];
mean_rad    = [3.75 4.25];
dev_rad     = [.5 1];
dev_center  = [80 100];
gauss_noise       = .2;
is_gauss_filt = 1;
T = 2;
m_phantom = scs_phantom_gen(dims, mean_rad, dev_rad, dev_center,is_gauss_filt,gauss_noise,T);
scs_slider_3dmatrix(m_phantom)