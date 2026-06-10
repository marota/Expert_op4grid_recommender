"""
tests/manoeuvre/test_plugins_interface.py
------------------------------------------
Couche d'algorithmes pluggables (``manoeuvre.plugins``) :

- registre par phase (natifs « libtopo », enregistrement de plugins, erreurs) ;
- façade ``PlanificateurTopologie`` : les trois phases A/B/C sur un poste réel
  (CARRIP3), composition A+B quand la phase C est désactivée ;
- type pivot ``CibleDetaillee`` (aller-retour graphe <-> états d'organes) ;
- vérification **indépendante** : verdicts recalculés par l'orchestrateur,
  y compris pour un plugin tiers mensonger ou émettant un organe inconnu.
"""

from __future__ import annotations

import pytest

from expert_op4grid_recommender.manoeuvre.topologie import (
    TopologieNodale,
    PosteTopologique,
)
from expert_op4grid_recommender.manoeuvre.algo.results import (
    Manoeuvre,
    ResultatManoeuvres,
)
from expert_op4grid_recommender.manoeuvre.plugins import (
    CibleDetaillee,
    IdentificateurTopologieDetaillee,
    PhaseNonConfiguree,
    PlanificateurNodal,
    PlanificateurTopologie,
    ResultatIdentification,
    ResultatPlanification,
    SequenceurManoeuvres,
    disponibles,
    get,
    register,
    verifier_sequence,
)
from expert_op4grid_recommender.manoeuvre.plugins import registry as registry_mod

from .fixture_loader import build_graph_from_fixture, list_available_fixtures


pytestmark = pytest.mark.skipif(
    "CARRIP3" not in list_available_fixtures(),
    reason="Fixture CARRIP3 absente.",
)

VL = "CARRIP3"

# Cible 2 nœuds réalisable sur la fixture (cf. test_carrip3_manoeuvre.py).
CIBLE_NOEUD_0 = [
    "BERT L31CARRI", "BRENOL31CARRI", "CARRIL31PERSA", "CARRIL31RANTI",
    "CARRIL31U.MON", "CARRIL31VALES", "CARRIL32U.MON",
]
CIBLE_NOEUD_1 = [
    "BARR6L31CARRI", "CARRI3T312", "CARRI3T313", "CARRI3T314",
    "CARRIL31V.PAU", "CARRIY631", "CARRIY632", "CARRIY633",
]


@pytest.fixture()
def poste() -> PosteTopologique:
    return PosteTopologique.from_graph(build_graph_from_fixture(VL), VL)


@pytest.fixture()
def topo_cible(poste: PosteTopologique) -> TopologieNodale:
    groupes = [list(CIBLE_NOEUD_0), list(CIBLE_NOEUD_1)]
    connectes = set(CIBLE_NOEUD_0) | set(CIBLE_NOEUD_1)
    for noeud in poste.topologie_nodale.noeuds.values():
        restes = noeud.equipment_ids - connectes
        if restes and not (noeud.equipment_ids & connectes):
            groupes.append(sorted(restes))
    return TopologieNodale.from_node_groups(VL, groupes)


# ---------------------------------------------------------------------------
# Registre
# ---------------------------------------------------------------------------

def test_libtopo_enregistre_pour_les_trois_phases():
    dispo = disponibles()
    assert set(dispo) == {"identificateur", "sequenceur", "planificateur"}
    for phase, noms in dispo.items():
        assert "libtopo" in noms, f"libtopo absent de la phase {phase}"


def test_get_instancie_et_respecte_les_contrats():
    assert isinstance(get("identificateur", "libtopo"),
                      IdentificateurTopologieDetaillee)
    assert isinstance(get("sequenceur", "libtopo"), SequenceurManoeuvres)
    assert isinstance(get("planificateur", "libtopo"), PlanificateurNodal)


def test_get_phase_ou_nom_inconnus():
    with pytest.raises(ValueError):
        get("phase_inexistante", "libtopo")
    with pytest.raises(KeyError):
        get("sequenceur", "algo_inexistant")


def test_register_refuse_le_doublon_silencieux():
    class Dummy:
        nom = "doublon_test"

        def sequencer(self, poste, cible, **options):  # pragma: no cover
            raise NotImplementedError

    register("sequenceur", "doublon_test", Dummy)
    try:
        with pytest.raises(ValueError, match="déjà enregistré"):
            register("sequenceur", "doublon_test", Dummy)
        register("sequenceur", "doublon_test", Dummy, remplacer=True)
    finally:
        registry_mod._registres["sequenceur"].pop("doublon_test", None)


# ---------------------------------------------------------------------------
# Type pivot CibleDetaillee
# ---------------------------------------------------------------------------

def test_cible_detaillee_aller_retour(poste: PosteTopologique):
    cible = CibleDetaillee.from_graph(poste.graph, VL)
    assert cible.etats_organes, "aucun organe capturé"
    assert cible.organes_inconnus(poste) == []
    # to_graph sur l'état capturé = topologie nodale identique, poste non muté.
    avant = poste.topologie_nodale.partition()
    G2 = cible.to_graph(poste)
    assert cible.topologie_nodale(poste).partition() == avant
    assert poste.topologie_nodale.partition() == avant
    assert G2 is not poste.graph
    # diff : bascule d'un organe -> détectée des deux côtés.
    sid = next(iter(cible.etats_organes))
    autre = CibleDetaillee(VL, dict(cible.etats_organes))
    autre.etats_organes[sid] = not autre.etats_organes[sid]
    assert set(cible.diff(autre)) == {sid}


# ---------------------------------------------------------------------------
# Façade : trois phases avec les algorithmes natifs
# ---------------------------------------------------------------------------

def test_phase_c_planifier_libtopo(poste, topo_cible):
    pipe = PlanificateurTopologie()
    plan = pipe.planifier(poste, topo_cible)
    assert isinstance(plan, ResultatPlanification)
    assert plan.is_verified, plan.message
    assert plan.nb_manoeuvres > 0
    # La cible détaillée jointe est bien l'état atteint par rejeu de la séquence
    # et réalise la partition nodale visée.
    assert plan.cible_detaillee is not None
    rejeu = CibleDetaillee.from_manoeuvres(poste, plan.manoeuvres)
    assert plan.cible_detaillee.diff(rejeu) == {}
    assert plan.cible_detaillee.topologie_nodale(poste).meme_topologie(topo_cible)


def test_phase_a_identifier_libtopo(poste, topo_cible):
    pipe = PlanificateurTopologie()
    ident = pipe.identifier_topologie_detaillee(poste, topo_cible)
    assert isinstance(ident, ResultatIdentification)
    assert ident.is_realisable, ident.message
    assert ident.cible is not None
    assert ident.cible.topologie_nodale(poste).meme_topologie(topo_cible)
    # Sous-produit : la séquence du planificateur natif est jointe.
    assert ident.sequence is not None and ident.sequence.is_verified


def test_phase_b_sequencer_libtopo(poste, topo_cible):
    pipe = PlanificateurTopologie()
    ident = pipe.identifier_topologie_detaillee(poste, topo_cible)
    seq = pipe.sequencer(poste, ident.cible, mode="smooth")
    assert seq.is_verified, seq.message
    assert seq.is_verified_detaillee, seq.ecarts
    # La cible détaillée peut aussi être passée en dict switch_id -> ouvert.
    seq2 = pipe.sequencer(poste, ident.cible.etats_organes, mode="smooth")
    assert seq2.is_verified_detaillee


def test_composition_a_plus_b_sans_planificateur(poste, topo_cible):
    pipe = PlanificateurTopologie(planificateur=None)
    plan = pipe.planifier(poste, topo_cible)
    assert plan.is_verified, plan.message
    assert plan.is_verified_detaillee, plan.sequence.ecarts
    assert plan.cible_detaillee is not None


def test_phase_manquante_leve(poste, topo_cible):
    pipe = PlanificateurTopologie(identificateur=None, sequenceur=None,
                                  planificateur=None)
    with pytest.raises(PhaseNonConfiguree):
        pipe.identifier_topologie_detaillee(poste, topo_cible)
    with pytest.raises(PhaseNonConfiguree):
        pipe.planifier(poste, topo_cible)
    cible = CibleDetaillee.from_graph(poste.graph, VL)
    with pytest.raises(PhaseNonConfiguree):
        pipe.sequencer(poste, cible)


# ---------------------------------------------------------------------------
# Plugins tiers : la vérification indépendante fait foi
# ---------------------------------------------------------------------------

class SequenceurMenteur:
    """Plugin tiers qui ne fait rien mais prétend avoir tout vérifié."""
    nom = "menteur"

    def sequencer(self, poste, cible, **options) -> ResultatManoeuvres:
        res = ResultatManoeuvres(
            voltage_level_id=poste.voltage_level_id,
            topo_initiale=poste.topologie_nodale,
            topo_cible=cible.topologie_nodale(poste),
        )
        res.is_verified = True            # mensonge
        res.is_verified_detaillee = True  # mensonge
        res.message = "tout va bien (dit le plugin)"
        return res


def test_verification_independante_demasque_un_plugin(poste, topo_cible):
    pipe = PlanificateurTopologie(sequenceur=SequenceurMenteur())
    ident = pipe.identifier_topologie_detaillee(poste, topo_cible)
    seq = pipe.sequencer(poste, ident.cible)
    # La cible exige des manœuvres : la séquence vide ne peut pas vérifier.
    assert not seq.is_verified
    assert not seq.is_verified_detaillee
    assert seq.topo_obtenue is not None
    assert seq.ecarts, "les écarts détaillés vs la cible doivent être consignés"


def test_verification_signale_un_organe_inconnu(poste):
    res = ResultatManoeuvres(
        voltage_level_id=VL,
        topo_initiale=poste.topologie_nodale,
        topo_cible=poste.topologie_nodale,
        manoeuvres=[Manoeuvre("ORGANE_FANTOME", "OPEN", "test")],
    )
    verifier_sequence(poste, res, topo_cible=poste.topologie_nodale)
    assert any("organe inconnu" in e and "ORGANE_FANTOME" in e
               for e in res.ecarts)


class PlanificateurConstant:
    """Plugin C minimal : délègue au natif, pour tester l'injection par nom."""
    nom = "constant_test"

    def planifier(self, poste, topo_cible, **options) -> ResultatPlanification:
        from expert_op4grid_recommender.manoeuvre.plugins import (
            LibTopoPlanificateur,
        )
        return LibTopoPlanificateur().planifier(poste, topo_cible, **options)


def test_plugin_enregistre_puis_resolu_par_nom(poste, topo_cible):
    register("planificateur", "constant_test", PlanificateurConstant)
    try:
        pipe = PlanificateurTopologie(planificateur="constant_test")
        plan = pipe.planifier(poste, topo_cible)
        assert plan.is_verified, plan.message
    finally:
        registry_mod._registres["planificateur"].pop("constant_test", None)
