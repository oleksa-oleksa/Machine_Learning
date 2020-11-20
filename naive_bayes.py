"""
There are Chris and Sara. They are writing E-mails only with 3 words: love, life, deal
Chris: love (0.1), life(0.1), deal(0.8)
Sara: love (0.3), life(0.3), deal(0.2)
P(Chris) = 0.5
P(Sara) = 0.5
"""

P_Chris = 0.5
P_Chris_love = 0.1
P_Chris_life = 0.1
P_Chris_deal = 0.8

P_Sara = 0.5
P_Sara_love = 0.5
P_Sara_life = 0.3
P_Sara_deal = 0.2

# Whom belongs "life deal" to?
print("Whom belongs love deal to?")
P_Chris_prior = P_Chris * P_Chris_life * P_Chris_deal
P_Sara_prior = P_Sara * P_Sara_life * P_Sara_deal
print("This is Chris: ", P_Chris_prior)
print("This is Sara:", P_Sara_prior)

print("Chris!") if P_Chris_prior > P_Sara_prior else print("Sara")

P_Chris_life_deal_posterior = P_Chris_prior / (P_Chris_prior + P_Sara_prior)
print("P_Chris_life_deal_posterior = ", P_Chris_life_deal_posterior)

P_Sara_life_deal_posterior = P_Sara_prior / (P_Chris_prior + P_Sara_prior)
print("P_Chris_life_deal_posterior = ", P_Sara_life_deal_posterior)


# Whom belongs "love deal" to?
print("Whom belongs love deal to?")
P_Chris_prior = P_Chris * P_Chris_love * P_Chris_deal
P_Sara_prior = P_Sara * P_Sara_love * P_Sara_deal
print("This is Chris: ", P_Chris_prior)
print("This is Sara:", P_Sara_prior)

print("Chris!") if P_Chris_prior > P_Sara_prior else print("Sara")

P_Chris_life_deal_posterior = P_Chris_prior / (P_Chris_prior + P_Sara_prior)
print("P_Chris_life_deal_posterior = ", P_Chris_life_deal_posterior)

P_Sara_life_deal_posterior = P_Sara_prior / (P_Chris_prior + P_Sara_prior)
print("P_Chris_life_deal_posterior = ", P_Sara_life_deal_posterior)

