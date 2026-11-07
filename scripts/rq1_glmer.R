# Mixed-effects logistic models for RQ1: random intercepts for source sentence
# and for each model output (sentence x perturbation x seed).
suppressMessages(library(lme4))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(file.path(args[1], "arcs_for_glmer.csv"))
d$perturbation <- factor(d$perturbation, levels = c("partialReverse", "localShuffle3", "fullShuffle"))
d$output <- paste(d$item, d$perturbation, d$seed)
ctl <- glmerControl(optimizer = "bobyqa", calc.derivs = FALSE)
rows <- list(); slopes <- list()
for (ds in unique(d$dataset)) {
  x <- d[d$dataset == ds, ]
  f0 <- is_recovered ~ log_dist + log_gold + sent_len_z + (1 | item) + (1 | output)
  m0 <- glmer(f0, x, family = binomial, nAGQ = 0, control = ctl)
  m1 <- update(m0, . ~ . + perturbation)
  m2 <- update(m1, . ~ . + log_dist:perturbation)
  a01 <- anova(m0, m1); a12 <- anova(m1, m2)
  rows[[ds]] <- data.frame(dataset = ds,
    chisq_pert = a01$Chisq[2], df_pert = a01$Df[2], p_pert = a01$`Pr(>Chisq)`[2],
    chisq_int = a12$Chisq[2], df_int = a12$Df[2], p_int = a12$`Pr(>Chisq)`[2],
    aic0 = AIC(m0), aic1 = AIC(m1), aic2 = AIC(m2),
    b_dist_m0 = fixef(m0)["log_dist"], sd_item = sqrt(VarCorr(m2)$item[1]), sd_output = sqrt(VarCorr(m2)$output[1]))
  b <- fixef(m2); V <- as.matrix(vcov(m2))
  for (p in levels(x$perturbation)) {
    nm <- "log_dist"
    if (p != "partialReverse") nm <- c(nm, paste0("log_dist:perturbation", p))
    est <- sum(b[nm]); se <- sqrt(sum(V[nm, nm]))
    slopes[[paste(ds, p)]] <- data.frame(dataset = ds, perturbation = p, slope = est, se = se,
                                         lo = est - 1.96 * se, hi = est + 1.96 * se)
  }
  cat(ds, "done\n")
}
write.csv(do.call(rbind, rows), file.path(args[1], "rq1_glmer_tests.csv"), row.names = FALSE)
write.csv(do.call(rbind, slopes), file.path(args[1], "rq1_glmer_slopes.csv"), row.names = FALSE)
