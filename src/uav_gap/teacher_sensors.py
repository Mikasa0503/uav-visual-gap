"""Transport delay for privileged measured state, with current known commands."""
from collections import deque


class TeacherObservationDelay:
    def __init__(self,steps):
        if type(steps) is not int or steps<0:
            raise ValueError('Delay must be a nonnegative integer')
        self.steps = steps
        self.history = None

    def __call__(self,observation):
        if observation.ndim!=2 or observation.shape[1]!=27:
            raise ValueError('Teacher observation must have 27 features')
        if self.history is None:
            self.history = deque([observation.clone() for _ in range(self.steps)],maxlen=self.steps+1)
        self.history.append(observation.clone())
        delayed = self.history.popleft()
        # Goal/velocity/rates/gravity and relative gap geometry are measurements.
        # The command sent by this controller is already known without transport.
        delayed[:,12:16] = observation[:,12:16]
        return delayed
