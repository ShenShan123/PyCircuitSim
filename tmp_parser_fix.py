            # Resolve modelcard path
            # For BSIM-CMG, try ASAP7 naming first (e.g., 7nm_TT.pm)
            # If using ASAP7 PDK, construct path with ASAP7 naming
            if self._asap7_modelcard_dir:
                # Use ASAP7 naming: nmos_lvt / nmos_lvt in modelcard
                asap7_name = f"{model_name}_lvt" if model_type == "nmos" else f"{model_name}_lvt"
                modelcard_path = Path(self._asap7_modelcard_dir) / f"{asap7_name}.pm"