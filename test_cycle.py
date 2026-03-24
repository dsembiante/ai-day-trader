from crew import run_trading_cycle
from circuit_breaker import CircuitBreaker

print('Starting test cycle...')
cb = CircuitBreaker()
run_trading_cycle(cb)
print('Test cycle complete')
