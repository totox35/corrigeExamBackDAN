package fr.istic.service.dto;

import io.quarkus.runtime.annotations.RegisterForReflection;
import java.io.Serializable;

/**
 * A DTO for the {@link fr.istic.domain.Prediction} entity.
 */
@RegisterForReflection
public class ResponseGroupDTO implements Serializable {

    public Long id;

    public Long questionId;

    public Long[] predictionIds;

    public Double[] averageEmbedding;


    @Override
    public boolean equals(Object o) {
        if (!(o instanceof ResponseGroupDTO)) {
            return false;
        }
        return id != null && id.equals(((ResponseGroupDTO) o).id);
    }

    @Override
    public int hashCode() {
        return 31;
    }

    @Override
    public String toString() {
        return "ResponseGroupeDTO{" +
            "id=" + id +
            ", questionId='" + questionId + "'" +
            ", predictionIds='" + predictionIds + "'" +
            ", averageEmbedding='" + averageEmbedding + "'" +
            "}";
    }
}
