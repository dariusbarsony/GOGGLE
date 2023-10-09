library(bnlearn)

# Download ECOLI70
download.file(
  "https://www.bnlearn.com/bnrepository/ecoli70/ecoli70.rda",
  "./ecoli70.rda"
)
# Load ECOLI70 bayesian betwork.
load("./ecoli70.rda")
# Set number of samples.
n <- 1000
# Sample from the network.
data <- rbn(bn, n)
# Write to CSV file.
write.csv(data, paste0("ecoli70_", n, ".csv"), row.names = FALSE)
