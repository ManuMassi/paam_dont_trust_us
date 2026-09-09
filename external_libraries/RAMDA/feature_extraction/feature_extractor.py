import networkx as nx
import os
import time
import logging
import io
import contextlib
from contextlib import contextmanager
from androguard.misc import AnalyzeAPK
from androguard.core.apk import APK

from logging import Logger
from utils import extract_apk_name

# Suppress verbose logging from androguard and its submodules
for _name in ("androguard", "androguard.core", "androguard.misc", "androguard.decompiler"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.CRITICAL)
    _lg.propagate = False
# Ensure global logging is enabled; only silence androguard-specific loggers above.
logging.disable(logging.NOTSET)


@contextmanager
def _suppress_output():
    """Context manager to suppress stdout and stderr at OS-level and Python-level."""
    devnull = os.open(os.devnull, os.O_RDWR)
    try:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(devnull)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)


def get_sensitive_apis(referred_sensitive_apis_dict: dict, call_graph_path: str):
    """
    Get sensitive APIs from call graph.
    :param referred_sensitive_apis_dict: a dict of sensitive APIs.
    :param call_graph_path: path of call graph.
    :return: a list of sensitive APIs.
    """
    sensitive_apis = [0] * len(referred_sensitive_apis_dict)
    if not os.path.exists(call_graph_path):
        raise FileNotFoundError(f"[RAMDA] Call graph file {call_graph_path} not found.")
    CG = nx.read_gml(call_graph_path)
    for node in CG.nodes:
        api_name = node
        if api_name in referred_sensitive_apis_dict:
            index = referred_sensitive_apis_dict[api_name]
            sensitive_apis[index] = 1

    return sensitive_apis


def get_permissions(referred_permissions_dict: dict, manifest_permission: list):
    """
    Get permissions from manifest.
    :param referred_permissions_dict: a dict of permissions.
    :param manifest_permission_path: path of manifest.
    :return: a list of permissions.
    """
    permissions = [0] * len(referred_permissions_dict)
    # if not os.path.exists(manifest_permission_path):
    #     raise FileNotFoundError(f"[RAMDA] Manifest permission file {manifest_permission_path} not found.")
    # with open(manifest_permission_path, 'r') as f:
    #     manifest_permissions = f.read().splitlines()

    for permission in manifest_permission:
        permission = permission.split(".")[-1]
        if permission in referred_permissions_dict:
            index = referred_permissions_dict[permission]
            permissions[index] = 1

    return permissions


def get_actions(referred_actions_dict: dict, intents: str):
    """
    Get actions from manifest.
    :param referred_actions_dict: a dict of actions.
    :param manifest_action_path: path of manifest.
    :return: a list of actions.
    """
    actions = [0] * len(referred_actions_dict)
    # if not os.path.exists(manifest_action_path):
    #     raise FileNotFoundError(f"[RAMDA] Manifest action file {manifest_action_path} not found.")

    # intents = json.load(open(manifest_action_path, 'r'))
    IntentList = []
    for _, value in intents.items():
        if value:
            for k, item_value in value.items():
                if k == "action":
                    IntentList.extend(item_value)

    for action in IntentList:
        action = action.split(".")[-1]
        if action in referred_actions_dict:
            index = referred_actions_dict[action]
            actions[index] = 1

    return actions


def extract_features(
    call_graph_path: str,
    permissions: str,
    intents: list,
    referred_sensitive_apis_dict: dict,
    referred_permissions_dict: dict,
    referred_actions_dict: dict,
    logger: Logger,
):
    """
    Extract features including 106 sensitive APIs, 147 permissions, and 126 intent actions.
    :param referred_sensitive_apis_dict: a dict of sensitive APIs.
    :param referred_permissions_dict: a dict of permissions.
    :param referred_actions_dict: a dict of actions.
    :param call_graph_path: path of call graph.
    :param manifest_permission_path: path of manifest.
    :param manifest_action_path: path of manifest.
    :param logger: logger.
    :return: a tuple of apk name and features.
    """
    apk_name = extract_apk_name(call_graph_path)
    tic = time.time()
    logger.info(f"Extracting features from {apk_name}")
    sensitive_apis = get_sensitive_apis(referred_sensitive_apis_dict, call_graph_path)
    permissions = get_permissions(referred_permissions_dict, permissions)
    actions = get_actions(referred_actions_dict, intents)
    features = sensitive_apis + permissions + actions

    return (apk_name, features, time.time() - tic)


def get_intent_filters_dict(apk: APK):
    """
    Get the intent filters.
    """
    items = {
        "activity": apk.get_activities(),
        "service": apk.get_services(),
        "receiver": apk.get_receivers(),
        "provider": apk.get_providers(),
    }
    intent_filters_dict = {}

    for item_type, item_list in items.items():
        for item in item_list:
            intent_filters = apk.get_intent_filters(item_type, item)
            intent_filters_dict[item] = intent_filters

    return intent_filters_dict


def extract_features1(
    apk_path: str,
    call_graph_dir: str,
    referred_sensitive_apis_dict: dict,
    referred_permissions_dict: dict,
    referred_actions_dict: dict,
    logger: Logger,  
):
    
    tic = time.time()
    try:
        with _suppress_output():
            a, _, _ = AnalyzeAPK(apk_path)
        app_name = extract_apk_name(apk_path)
        callgraph_file = os.path.join(call_graph_dir, app_name + ".gml.gz")
        intents = get_intent_filters_dict(a)
        permissions = a.get_permissions()

        sensitive_apis = get_sensitive_apis(referred_sensitive_apis_dict, callgraph_file)
        permissions = get_permissions(referred_permissions_dict, permissions)
        actions = get_actions(referred_actions_dict, intents)
        features = sensitive_apis + permissions + actions

        return (app_name, apk_path, features, time.time() - tic)
    except Exception as e:
        if logger:
            logger.error(f"Error processing {apk_path}: {e}")
        else:
            print(f"Error processing {apk_path}: {e}")
        return (None, apk_path, None, time.time() - tic)