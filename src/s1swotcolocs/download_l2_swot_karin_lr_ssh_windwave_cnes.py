import os
import argparse
import logging
import datetime
import paramiko
import re
from collections import defaultdict
from tqdm import tqdm
from stat import S_ISDIR

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def is_dir(sftp, path):
    try:
        return S_ISDIR(sftp.stat(path).st_mode)
    except IOError:
        return False


def extract_date_from_filename(filename):
    """
    Extrait la date de début du format:
    SWOT_L2_LR_SSH_WindWave_032_268_20250507T134734_...
    Retourne un objet datetime.date
    """
    try:
        # Recherche d'une chaine de 8 chiffres suivie d'un T (YYYYMMDDT...)
        match = re.search(r"_(\d{8})T", filename)
        if match:
            return datetime.datetime.strptime(match.group(1), "%Y%m%d").date()
    except Exception:
        return None
    return None


def download_with_progress(sftp, remote_path, local_path, filename):
    """
    Download a file with a progress bar

    Args:
        sftp (paramiko.SFTPClient): SFTP client
        remote_path (str): Path to the remote file
        local_path (str): Path to save the local file
        filename (str): Name of the file being downloaded (for display)

    Returns:
        None

    """
    file_size = sftp.stat(remote_path).st_size
    with tqdm(
        total=file_size, unit="B", unit_scale=True, desc=filename, leave=False
    ) as pbar:

        def cb(transferred, total):
            pbar.update(transferred - pbar.n)

        sftp.get(remote_path, local_path, callback=cb)


def main():
    parser = argparse.ArgumentParser(
        description="Téléchargeur SWOT L2 par plage de dates"
    )
    parser.add_argument("--user", required=True, help="Login SFTP (ex: email)")
    parser.add_argument("--password", required=True, help="Mot de passe")
    parser.add_argument("--dest", required=True, help="Répertoire de destination local")
    parser.add_argument("--start", required=True, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Date de fin (YYYY-MM-DD)")
    parser.add_argument(
        "--host",
        default="ftp-access.aviso.altimetry.fr",
        help="Hôte SFTP [optional défaut: ftp-access.aviso.altimetry.fr]",
    )
    parser.add_argument(
        "--port", type=int, default=2221, help="Port SFTP [optional default: 2221]"
    )
    parser.add_argument(
        "--productID",
        choices=["PID0", "PGD0"],
        default="PID0",
        help="ID du produit SWOT L2 (PID0 ou PGD0) [optionnel défaut: PID0]",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Activer le mode verbeux"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exécuter en mode test sans téléchargement",
    )
    args = parser.parse_args()

    BASE_PATH = f"/swot_products/l2_karin/l2_lr_ssh/{args.productID}/WindWave"
    logger.info(f"Produit sélectionné: {args.productID}")
    logger.info(f"Répertoire distant de base: {BASE_PATH}")
    logger.info(f"Plage de dates: {args.start} à {args.end}")
    logger.info(f"Répertoire local de destination: {args.dest}")
    logger.info(f"Mode test (dry-run): {'Activé' if args.dry_run else 'Désactivé'}")
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    try:
        start_dt = datetime.datetime.strptime(args.start, "%Y-%m-%d").date()
        end_dt = datetime.datetime.strptime(args.end, "%Y-%m-%d").date()
    except ValueError:
        logger.error("Format date invalide. Utilisez YYYY-MM-DD")
        return

    os.makedirs(args.dest, exist_ok=True)

    stats0 = {"downloaded": 0, "errors": 0, "already_exists": 0, "filtered": 0}
    stats = defaultdict(int)
    for uu in stats0:
        stats[uu] = stats0[uu]

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        logger.info(f"Connexion à {args.host}...")
        ssh.connect(
            args.host, port=args.port, username=args.user, password=args.password
        )
        sftp = ssh.open_sftp()

        logger.info(f"Exploration de {BASE_PATH} ...")
        # Liste tous les dossiers de cycles
        try:
            all_items = sftp.listdir(BASE_PATH)
            cycles = [d for d in all_items if d.startswith("cycle_")]
            cycles.sort()  # Optionnel: pour traiter dans l'ordre chronologique
        except IOError:
            logger.error(f"Impossible de lire le répertoire racine {BASE_PATH}")
            return

        for cci, cycle_dir in enumerate(cycles):
            remote_cycle_path = f"{BASE_PATH}/{cycle_dir}"
            logger.info(
                f"Analyse du dossier {cycle_dir} number {cci+1} sur {len(cycles)}"
            )

            try:
                files = sftp.listdir(remote_cycle_path)
                for filename in files:
                    if not filename.endswith(".nc"):
                        continue

                    file_date = extract_date_from_filename(filename)

                    if file_date and start_dt <= file_date <= end_dt:
                        remote_file_path = f"{remote_cycle_path}/{filename}"
                        local_file_path = os.path.join(args.dest, filename)

                        if os.path.exists(local_file_path):
                            stats["already_exists"] += 1
                            continue

                        try:
                            if args.dry_run:
                                logger.info(f"[DRY-RUN] Prêt à télécharger: {filename}")
                                stats["supposed-to-be-downloaded"] += 1
                            else:
                                download_with_progress(
                                    sftp, remote_file_path, local_file_path, filename
                                )
                                stats["downloaded"] += 1
                        except Exception as e:
                            logger.error(f"Erreur sur {filename}: {e}")
                            stats["errors"] += 1
                    else:
                        stats["filtered"] += 1
            except Exception as e:
                logger.warning(f"Erreur d'accès au cycle {cycle_dir}: {e}")

        sftp.close()
        ssh.close()

    except Exception as e:
        logger.error(f"Erreur globale : {e}")

    logger.info("=========================================")
    logger.info(f" Fin du traitement pour {args.start} à {args.end}")
    logger.info(f" - Téléchargés          : {stats['downloaded']}")
    logger.info(f" - Déjà présents        : {stats['already_exists']}")
    logger.info(f" - Hors plage de dates  : {stats['filtered']}")
    logger.info(f" - Erreurs              : {stats['errors']}")
    logger.info(f" - tag to get (dry-run) : {stats['supposed-to-be-downloaded']}")
    logger.info("=========================================")


if __name__ == "__main__":
    main()
