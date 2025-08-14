Sys.setenv(PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig")
install.packages('Matrix')
install.packages("epiR")
install.packages('planningML')
install.packages('pmsampsize')

library('epiR')

epi.sssimpleestb(N = NA, Py, epsilon, error = "relative", se, sp, nfractional = FALSE, conf.level = 0.95)



