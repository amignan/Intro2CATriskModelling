# Title: CAT Risk Modelling Starter-Kit (in R)
# Author: Arnaud Mignan
# Date: 01.02.2024
# Description: A basic template to develop a catastrophe (CAT) risk model (here with ad-hoc parameters & models).
# License: MIT
# Version: 1.1
# Dependencies: ggplot2, gridExtra, lattice
# Contact: arnaud@mignanriskanalytics.com
# Citation: Mignan, A. (2025), Introduction to Catastrophe Risk Modelling – A Physics-based Approach. Cambridge University Press, DOI: 10.1017/9781009437370

#Copyright 2024 A. Mignan
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), 
#to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
#and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
#IN THE SOFTWARE.


library(ggplot2)
library(gridExtra)
library(lattice)


# ad-hoc function
func_src2ev <- function(src_S) {
  # The maximum size S of an event (ev_Smax) is constrained by the source from which it originates,
  # with src_S the source characteristic from which ev_Smax can be inferred - See model examples in Section 2.2.
  c <- 0.1
  k <- 2
  ev_Smax <- c * src_S^k     # ad-hoc model
  return(ev_Smax)            # maximum-size event (or characteristic event for the source)
}

func_intensity <- function(S, r) {
  # The impact of each event on the environment is assessed by the hazard intensity I(x,y) across a
  # geographical grid (x,y). A static footprint is defined as a function of distance from source r(x,y)
  # and event size S - See model examples in Section 2.4.
  c0 <- -1
  c1 <- 0.1
  c2 <- 1
  c3 <- 5
  I <- exp(c0 + c1 * S - c2 * log(r + c3))  # ad-hoc model
  return(I)
}

erf <- function(x) return(2 * pnorm(x * sqrt(2)) - 1)
func_vulnerability <- function(I, Theta_nu) {
  # The vulnerability curve expresses the damage D (or Mean Damage Ratio, MDR) expected on an asset of
  # characteristics Theta_nu given a hazard intensity load I - See model examples in Section 3.2.5.
  MDR <- 0.5 + 0.5 * erf((log(I) - Theta_nu$mu) / sqrt(2 * Theta_nu$sigma^2))   # here cum. lognormal distr.
  return(MDR)
}


# ad-hoc parameters (to be replaced by peril- & region-specific values)
xmin <- 0        # [km]
xmax <- 100
dx <- 1
ymin <- 0        # [km]
ymax <- 100
dy <- 1
src1_x0 <- 25    # source 'src1' of coordinates (x0,y0) and size S
src1_y0 <- 25
src1_S <- 5
src2_x0 <- 75
src2_y0 <- 50
src2_S <- 8
Smin <- 0.1      # minimum event size [ad hoc unit] (e.g., energy)
dS_log10 <- 0.1  # event size increment (in log10 scale)
a <- 0.5         # for Eq. 2.38: log10(rate_cum(S)) = a - b log10(S); see some values in Tab. 2.5
b <- 1
Theta_nu <- list(mu = log(.04), sigma = .1) # for Eq. 3.4: cum. lognormal distr.; see some values in Section 3.2.5

# define environment (i.e., grid for footprints)
x = seq(xmin, xmax, dx)
y = seq(ymin, ymax, dy)
grid <- expand.grid(x = x, y = y)

# exposure footprint defined below as a square town of uniform asset value 1
padW <- 30
padE <- 25
padN <- 40
padS <- 20
nu_grid <- array(1, dim = c(length(grid$x)))
for (i in 1:length(grid$x)) {
  if (grid$x[i] < padW){nu_grid[i] = 0}
  if (grid$x[i] >= length(x)-padE){nu_grid[i] = 0}
  if (grid$y[i] < padS){nu_grid[i] = 0}
  if (grid$y[i] >= length(y)-padN){nu_grid[i] = 0}
}


## HAZARD ASSESSMENT ##
# define source model (here, 2 point sources)
Src <- data.frame(ID = c('src1', 'src2'), x0 = c(src1_x0, src2_x0), y0 = c(src1_y0, src2_y0), S = c(src1_S, src2_S))


# define size distribution
Smax1 <- func_src2ev(src1_S)
Smax2 <- func_src2ev(src2_S)
Smax <- max(Smax1, Smax2)
Si <- 10^seq(log10(Smin), log10(Smax), dS_log10)
Si1 <- 10^seq(log10(Smin), log10(Smax1), dS_log10)
Si2 <- 10^seq(log10(Smin), log10(Smax2), dS_log10)
N1 <- length(Si1)
N2 <- length(Si2)
ratei <- 10^(a - b * (log10(Si) - dS_log10 / 2)) - 10^(a - b * (log10(Si) + dS_log10 / 2))  # e.g., Eq. 2.65


# define event table
# peril ID: e.g., EQ, VE, AI... Tab. 1.7
EventTable <- data.frame(ID = paste0('ID', 1:(N1 + N2)), Src = rep(c('src1', 'src2'), c(N1, N2)), S = c(Si1, Si2),rate = c(ratei[1:N1], ratei[1:N2]))

# correct rate, which is function of the stochastic set definition:
# in the present case, we have two sources with equal share of the overall event activity defined by (a,b)
Nevent_perSi = c(rep(2, 2 * N1), rep(1, N2-N1)) # if N1 < N2 (i.e., src1_S < src2_S)
EventTable$rate = EventTable$rate / Nevent_perSi
# Whichever stochastic construct, we must have EventTable['rate'].sum() = np.sum(ratei)


# define intensity I grid footprint catalog
I_grid <- array(0, dim = c(N1 + N2, length(grid$x)))
for (i in 1:(N1 + N2)) {
  ind <- which(Src$ID == EventTable$Src[i])[1]
  for (j in 1:length(grid$x)){
    r_grid <- sqrt((grid$x[j] - Src$x0[ind])^2 + (grid$y[j] - Src$y0[ind])^2)
    I_grid[i,j] <- func_intensity(EventTable$S[i], r_grid)   
  }
}

# -> calculate hazard metrics (see solution to exercise #2.4)


## RISK ASSESSMENT ##
# calculate damage D grid & loss L grid footprints
D_grid <- array(0, dim = c(N1 + N2, length(grid$x)))
L_grid <- array(0, dim = c(N1 + N2, length(grid$x)))
for (i in 1:(N1 + N2)) {
  D_grid[i,] <- func_vulnerability(I_grid[i,], Theta_nu)
  L_grid[i,] <- D_grid[i,] * nu_grid
}

# update event table as loss table
ELT <- EventTable
ELT$loss <- sapply(1:(N1 + N2), function(i) sum(L_grid[i,]))

# -> calculate risk metrics (see solution to exercise #3.2)


## plot templates ##
pdf('plots_template_R.pdf', width = 20, height = 15)

df_plot <- data.frame(src_Si = seq(0.1, 10, length.out = 100), func_src2ev = func_src2ev(seq(0.1, 10, length.out = 100)))
plot1 <- ggplot(df_plot, aes(x = src_Si, y = func_src2ev)) +
  geom_line(color = 'black') +
  geom_hline(yintercept = Smin, linetype = 'dashed', color = 'black') +
  geom_vline(xintercept = src1_S, linetype = 'dashed', color = 'orange') +
  geom_vline(xintercept = src2_S, linetype = 'dashed', color = 'red') +
  labs(x = 'Source size src_S', y = 'Max. event size Smax', title = 'Characteristic event size') +
  theme_minimal()

df_rate <- data.frame(Si = Si, ratei = ratei)
ind1 <- which(EventTable$Src == 'src1')
ind2 <- which(EventTable$Src == 'src2')

df_src1 <- EventTable[ind1, ]
df_src2 <- EventTable[ind2, ]
offset <- 1.1  # to avoid overlap
df_src2$rate <- df_src2$rate * offset
plot2 <- ggplot() +
  geom_line(data = df_rate, aes(x = Si, y = ratei), color = 'black') +
  geom_point(data = df_src1, aes(x = S, y = rate), color = 'orange') +
  geom_point(data = df_src2, aes(x = S, y = rate), color = 'red') +
  scale_x_log10() +
  scale_y_log10() +
  labs(x = 'Event size S', y = 'Rate', title = 'Stochastic set distribution') +
  theme_minimal() +
  theme(legend.position = 'top')

df_intensity <- data.frame(
  ri = seq(0, 50, 0.1),
  intensity_src1 = func_intensity(Smax1, seq(0, 50, 0.1)),
  intensity_src2 = func_intensity(Smax2, seq(0, 50, 0.1))
)
plot3 <- ggplot(df_intensity, aes(x = ri)) +
  geom_line(aes(y = intensity_src1), color = 'orange') +
  geom_line(aes(y = intensity_src2), color = 'red') +
  labs(x = 'Distance from source r', y = 'Intensity I', title = 'Hazard intensity model') +
  theme_minimal() +
  theme(legend.position = 'top')

Ii = seq(.001, .1, .001)
df_vulnerability <- data.frame(Ii = Ii, mean_damage_ratio = func_vulnerability(Ii, Theta_nu))
plot4 <- ggplot(df_vulnerability, aes(x = Ii, y = mean_damage_ratio)) +
  geom_line(color = 'black') +
  labs(x = 'Intensity I', y = 'Mean damage ratio', title = 'Vulnerability curve') +
  theme_minimal()

df_nu = data.frame(nu = nu_grid, x = grid$x, y = grid$y)
plot5 <- ggplot() +
  geom_raster(data = df_nu, aes(x = x, y = y, fill = nu)) +
  scale_fill_gradient(low = "white", high = "darkgreen") +
  labs(x = 'x', y = 'y', title = 'Exposure footprint') +
  theme_minimal() +
  theme(legend.position="none")

df_I = data.frame(nu = I_grid[tail(ind1, 1),], x = grid$x, y = grid$y)
plot6 <- ggplot() +
  geom_raster(data = df_I, aes(x = x, y = y, fill = nu)) +
  geom_point(data = Src[Src$ID == 'src1',], aes(x = x0, y = y0), pch = 3, col = 'black')  +
  scale_fill_gradient(low = "white", high = "red") +
  labs(x = 'x', y = 'y', title = 'Hazard intensity footprint (Smax1)') +
  theme_minimal() +
  theme(legend.position="none")


df_D = data.frame(nu = D_grid[tail(ind1, 1),], x = grid$x, y = grid$y)
plot7 <- ggplot() +
  geom_raster(data = df_D, aes(x = x, y = y, fill = nu)) +
  geom_point(data = Src[Src$ID == 'src1',], aes(x = x0, y = y0), pch = 3, col = 'black')  +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(x = 'x', y = 'y', title = 'Expected damage footprint (Smax1)') +
  theme_minimal() +
  theme(legend.position="none")

df_L = data.frame(nu = L_grid[tail(ind1, 1),], x = grid$x, y = grid$y)
plot8 <- ggplot() +
  geom_raster(data = df_L, aes(x = x, y = y, fill = nu)) +
  geom_point(data = Src[Src$ID == 'src1',], aes(x = x0, y = y0), pch = 3, col = 'black')  +
  scale_fill_gradient(low = "white", high = "purple") +
  labs(x = 'x', y = 'y', title = 'Loss footprint (Smax1)') +
  theme_minimal() +
  theme(legend.position="none")

plot9 <- ggplot() +
  geom_raster(data = df_nu, aes(x = x, y = y, fill = nu)) +
  scale_fill_gradient(low = "white", high = "darkgreen") +
  labs(x = 'x', y = 'y', title = 'Exposure footprint') +
  theme_minimal() +
  theme(legend.position="none")

df_I = data.frame(nu = I_grid[tail(ind2, 1),], x = grid$x, y = grid$y)
plot10 <- ggplot() +
  geom_raster(data = df_I, aes(x = x, y = y, fill = nu)) +
  geom_point(data = Src[Src$ID == 'src2',], aes(x = x0, y = y0), pch = 3, col = 'black')  +
  scale_fill_gradient(low = "white", high = "red") +
  labs(x = 'x', y = 'y', title = 'Hazard intensity footprint (Smax2)') +
  theme_minimal() +
  theme(legend.position="none") +
  coord_fixed()

df_D = data.frame(nu = D_grid[tail(ind2, 1),], x = grid$x, y = grid$y)
plot11 <- ggplot() +
  geom_raster(data = df_D, aes(x = x, y = y, fill = nu)) +
  geom_point(data = Src[Src$ID == 'src2',], aes(x = x0, y = y0), pch = 3, col = 'black')  +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(x = 'x', y = 'y', title = 'Expected damage footprint (Smax2)') +
  theme_minimal() +
  theme(legend.position="none") +
  coord_fixed()

df_L = data.frame(nu = L_grid[tail(ind2, 1),], x = grid$x, y = grid$y)
plot12 <- ggplot() +
  geom_raster(data = df_L, aes(x = x, y = y, fill = nu)) +
  geom_point(data = Src[Src$ID == 'src2',], aes(x = x0, y = y0), pch = 3, col = 'black')  +
  scale_fill_gradient(low = "white", high = "purple") +
  labs(x = 'x', y = 'y', title = 'Loss footprint (Smax2)') +
  theme_minimal() +
  theme(legend.position="none") +
  coord_fixed()

fig <- grid.arrange(plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9, plot10, plot11, plot12, ncol = 4, nrow = 3)
dev.off()


