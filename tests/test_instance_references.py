"""Private reference classes preserve ownership without retaining stale owners."""

import copy
import gc
import weakref

import torch

import dendra as dn
from dendra.models.mod import pas
from dendra.models.parametric import Referency, add_instance_property


class _Owner:
    def __init__(self, value):
        self.t = torch.tensor(value, dtype=torch.float64)


def test_rebinding_references_preserves_callbacks_and_source_class():
    calls = []
    source = Referency()
    original_owner = _Owner(1.0)
    source.setreference("t", lambda: original_owner.t)
    source.setreference("probe", lambda: calls.append("probe") or 42)
    source_type = type(source)
    original_depth = len(source_type.__mro__)
    copied = copy.deepcopy(source)

    for value in range(2, 8):
        owner = _Owner(value)
        copied.setreference("t", lambda owner=owner: owner.t)
        assert copied.t is owner.t
        assert source.t is original_owner.t
        assert type(source) is source_type
        assert len(type(copied).__mro__) == original_depth
        assert calls == []

    assert copied.probe == 42
    assert source.probe == 42
    assert calls == ["probe", "probe"]


def test_rebinding_mechanism_references_preserves_declarations_and_depth():
    population = dn.SingleCompartment(N=1, C=1, dtype=torch.float64)
    population.insert(pas)
    population.build()
    source = population.mech.pas
    calls = []
    source.setreference("probe", lambda: calls.append("probe") or source.g)
    source_type = type(source)
    copied = copy.deepcopy(source)

    for value in range(2, 8):
        owner = _Owner(value)
        copied.setreference("t", lambda owner=owner: owner.t)
        assert copied.t is owner.t
        assert source.t is population.t
        assert type(source) is source_type
        assert len(type(copied).__mro__) == len(source_type.__mro__)
        assert type(copied)._range == source_type._range
        assert calls == []

    assert copied.probe is source.g
    assert calls == ["probe"]


def test_rebinding_reference_retains_an_authored_subclass():
    source = Referency()
    source.setreference("t", lambda: 1)
    source_type = type(source)

    class Authored(source_type):
        def doubled_time(self):
            return self.t * 2

    authored = Authored()
    authored.setreference("t", lambda: 3)

    assert isinstance(authored, Authored)
    assert authored.doubled_time() == 6
    assert type(authored).__bases__ == (Authored,)
    assert type(source) is source_type
    assert source.t == 1


def test_rebinding_reference_preserves_attributes_added_to_generated_class():
    source = Referency()
    source.setreference("t", lambda: 1)
    source_type = type(source)
    source_type.authored_setting = object()
    copied = copy.deepcopy(source)

    copied.setreference("t", lambda: 2)

    assert copied.authored_setting is source_type.authored_setting
    assert type(copied).__bases__ == (source_type,)
    assert source.t == 1
    assert copied.t == 2


def test_rebinding_reference_preserves_in_place_declaration_edits():
    population = dn.SingleCompartment(N=1, C=1, dtype=torch.float64)
    population.insert(pas)
    population.build()
    source = population.mech.pas
    source_type = type(source)
    source_type._range["g"] = 0.375
    copied = copy.deepcopy(source)
    owner = _Owner(2.0)

    copied.setreference("t", lambda: owner.t)

    assert type(copied).__bases__ == (source_type,)
    assert type(copied)._range["g"] == 0.375
    assert source_type._range["g"] == 0.375
    assert type(source) is source_type
    assert copied.t is owner.t


def test_reference_registry_does_not_retain_callback_owner_or_proxy_class():
    owner = _Owner(1.0)
    recipient = Referency()
    owner.recipient = recipient
    recipient.setreference("t", lambda owner=owner: owner.t)
    owner_ref = weakref.ref(owner)
    recipient_ref = weakref.ref(recipient)
    proxy_ref = weakref.ref(type(recipient))

    del owner, recipient
    gc.collect()

    assert owner_ref() is None
    assert recipient_ref() is None
    assert proxy_ref() is None


def test_reference_registry_does_not_retain_generated_metadata_instance_type():
    class SelfMetadata:
        def __init_subclass__(cls):
            super().__init_subclass__()
            cls.token = object.__new__(cls)

    recipient = SelfMetadata()
    recipient.value = 42
    add_instance_property(
        recipient, "answer", lambda recipient=recipient: recipient.value
    )
    recipient_ref = weakref.ref(recipient)
    proxy_ref = weakref.ref(type(recipient))
    token_ref = weakref.ref(type(recipient).token)
    assert recipient.answer == 42

    del recipient
    gc.collect()

    assert recipient_ref() is None
    assert proxy_ref() is None
    assert token_ref() is None


def test_reference_registry_accepts_unhashable_metaclasses():
    class UnhashableMeta(type):
        __hash__ = None

    class Source(metaclass=UnhashableMeta):
        pass

    source = Source()
    add_instance_property(source, "value", lambda: 1)
    depth = len(type(source).__mro__)
    add_instance_property(source, "other", lambda: 3)
    add_instance_property(source, "value", lambda: 2)

    assert source.value == 2
    assert source.other == 3
    assert isinstance(source, Source)
    assert len(type(source).__mro__) == depth


def test_reference_registry_uses_class_identity_without_metaclass_dispatch():
    calls = []

    class EqualMeta(type):
        def __hash__(cls):
            calls.append(("hash", id(cls)))
            return 42

        def __eq__(cls, other):
            calls.append(("eq", id(cls), id(other)))
            return True

    class First(metaclass=EqualMeta):
        pass

    class Second(metaclass=EqualMeta):
        pass

    first, second = First(), Second()
    add_instance_property(first, "value", lambda: 1)
    add_instance_property(second, "value", lambda: 2)
    add_instance_property(first, "value", lambda: 3)
    add_instance_property(second, "value", lambda: 4)

    assert type(first).__bases__[0] is First
    assert type(second).__bases__[0] is Second
    assert first.value == 3
    assert second.value == 4
    assert calls == []


def test_generated_metadata_fingerprint_does_not_dispatch_metaclass_equality():
    calls = []

    class EqualMeta(type):
        def __eq__(cls, other):
            calls.append((id(cls), id(other)))
            return True

        __hash__ = None

    class SelfMetadata(metaclass=EqualMeta):
        def __init_subclass__(cls):
            super().__init_subclass__()
            cls.token = object.__new__(cls)

    recipient = SelfMetadata()
    add_instance_property(recipient, "value", lambda: 1)
    depth = len(type(recipient).__mro__)
    add_instance_property(recipient, "value", lambda: 2)

    assert recipient.value == 2
    assert len(type(recipient).__mro__) == depth
    assert calls == []
