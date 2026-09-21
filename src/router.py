from src.rollout import use_model
from src.shadow import log_shadow
from src.predictor import score

def decide(applicant, current_decision_fn, rollout_pct=0.0):
    model_decision = score(applicant)
    if use_model(rollout_pct):
        final = {'limit': model_decision['limit'], 'source': 'model'}
    else:
        final = {'limit': current_decision_fn(applicant), 'source': 'current'}
    log_shadow(applicant, model_decision, final)
    return final

if __name__ == '__main__':
    def current(a):
        return 5000
    print(decide({'age': 35, 'credit_amount': 8000, 'duration': 24,
                  'employment': '1<=X<4', 'purpose': 'radio/tv',
                  'checking_status': '0<=X<200', 'savings_status': '<100'},
                 current, rollout_pct=1.0))
