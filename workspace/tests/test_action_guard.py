import pytest
from physicalai_vla.action_guard import ActionGuard,CODEC,require_hardware_readiness


def ready():
 g=ActionGuard('fixed-model');g.reset('episode',0,100.)
 return g


def command(**kwargs):
 return dict(episode='episode',sequence=1,observation_time=100.,codec=CODEC,bundle_hash='fixed-model',dt=.05,actions=[0.]*7,now=100.1,**kwargs)


def test_valid_command_and_duplicate_stop_latches():
 g=ready();assert g.accept(**command())==[0.]*7
 with pytest.raises(ValueError,match='duplicate'):g.accept(**command())
 assert g.fault
 with pytest.raises(ValueError,match='latched'):g.accept(**{**command(),'sequence':2})
 g.reset('next',4,101.);assert not g.fault


@pytest.mark.parametrize('change',[{'episode':'old'},{'sequence':2},{'codec':'absolute10'},{'bundle_hash':'wrong'},{'dt':0.},
 {'actions':[0.]*10},{'actions':[float('nan')]*7},{'actions':[2.]*7},{'observation_time':99.},{'now':103.},{'now':99.}])
def test_invalid_commands_latch_fault(change):
 g=ready()
 with pytest.raises(ValueError):g.accept(**{**command(),**change})
 assert g.fault


def test_watchdog_and_hardware_commissioning_gate():
 g=ready();assert not g.watchdog(101.);assert g.watchdog(103.)
 with pytest.raises(ValueError,match='commissioning'):require_hardware_readiness({'mode':'hardware'})
