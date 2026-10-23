"""Asymmetric patient-transition fixture for the posthoc analyzer."""
from __future__ import annotations

import numpy as np


def patient_transition(left_y, left_pred, right_y, right_pred):
    left_y=np.asarray(left_y,dtype=np.int64);left_pred=np.asarray(left_pred,dtype=np.int64)
    right_y=np.asarray(right_y,dtype=np.int64);right_pred=np.asarray(right_pred,dtype=np.int64)
    if not (left_y.shape==left_pred.shape==right_y.shape==right_pred.shape):
        raise AssertionError("Patient arrays must have identical shapes")
    if not np.array_equal(left_y,right_y):raise AssertionError("Patient labels differ")
    y=left_y
    corrected=(left_pred==y)&(right_pred!=y)
    newly_wrong=(left_pred!=y)&(right_pred==y)
    return {"left_corrects_right_errors":int(corrected.sum()),
            "left_introduces_errors":int(newly_wrong.sum()),
            "net_correct_change_left_minus_right":int(corrected.sum()-newly_wrong.sum()),
            "ASD_net":int(np.sum(corrected&(y==0))-np.sum(newly_wrong&(y==0))),
            "TD_net":int(np.sum(corrected&(y==1))-np.sum(newly_wrong&(y==1)))}


def run_tests():
    # Deliberately asymmetric, with both classes represented in both directions.
    y=np.asarray([0,0,0,1,1,1])
    left=np.asarray([0,1,0,1,0,1])
    right=np.asarray([1,1,0,0,1,1])
    actual=patient_transition(y,left,y,right)
    expected={"left_corrects_right_errors":2,"left_introduces_errors":1,
              "net_correct_change_left_minus_right":1,"ASD_net":1,"TD_net":0}
    assert actual==expected,(actual,expected)
    reverse=patient_transition(y,right,y,left)
    assert reverse["left_corrects_right_errors"]==actual["left_introduces_errors"]
    assert reverse["left_introduces_errors"]==actual["left_corrects_right_errors"]
    assert reverse["net_correct_change_left_minus_right"]==-actual["net_correct_change_left_minus_right"]
    assert reverse["ASD_net"]==-actual["ASD_net"]
    assert reverse["TD_net"]==-actual["TD_net"]
    return {"passed":True,"non_symmetric_fixture":actual,"reverse_fixture":reverse}


if __name__=="__main__":
    print(run_tests())
